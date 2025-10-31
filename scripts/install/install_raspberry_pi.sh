#!/bin/bash
# Raspberry Pi Optimized Installation Script
# Doorbell Security System - Cross-Platform Installer

set -e

echo "🍓 Installing Doorbell Security System on Raspberry Pi..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    print_status "Checking if running on Raspberry Pi..."
    
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model)
        if [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_success "Detected: $PI_MODEL"
            return 0
        fi
    fi
    
    print_warning "Not running on Raspberry Pi - some features may not work"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    return 1
}

# Check available memory
check_memory() {
    print_status "Checking available memory..."
    
    MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEMORY_MB=$((MEMORY_KB / 1024))
    
    print_status "Available memory: ${MEMORY_MB}MB"
    
    if [[ $MEMORY_MB -lt 512 ]]; then
        print_error "Insufficient memory (< 512MB). This installation may fail."
        print_status "Consider using a Raspberry Pi with at least 1GB RAM"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    elif [[ $MEMORY_MB -lt 1024 ]]; then
        print_warning "Low memory detected. Using optimized installation..."
        export PIP_NO_CACHE_DIR=1
        export MAX_JOBS=1
        LOW_MEMORY=true
    else
        print_success "Sufficient memory available"
        LOW_MEMORY=false
    fi
}

# Configure swap for low memory systems
configure_swap() {
    if [[ "$LOW_MEMORY" == true ]]; then
        print_status "Configuring swap for low memory system..."
        
        # Check current swap
        SWAP_SIZE=$(free -m | grep Swap | awk '{print $2}')
        
        if [[ $SWAP_SIZE -lt 1024 ]]; then
            print_status "Increasing swap size to 2GB..."
            
            # Increase swap (if possible)
            if [ -f /etc/dphys-swapfile ]; then
                sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
                sudo dphys-swapfile swapoff
                sudo dphys-swapfile setup
                sudo dphys-swapfile swapon
                print_success "Swap size increased"
            else
                print_warning "Could not configure swap automatically"
            fi
        fi
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y || print_warning "Some packages may need manual upgrade"
    
    print_success "System updated"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies (optimized for Pi)..."
    
    # Essential packages
    sudo apt install -y \
        build-essential \
        cmake \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        curl \
        wget \
        || print_error "Failed to install essential packages"
    
    # Try to install system Python packages first (faster on Pi)
    print_status "Installing Python packages from apt (faster)..."
    sudo apt install -y \
        python3-numpy \
        python3-opencv \
        python3-pil \
        python3-requests \
        python3-yaml \
        || print_warning "Some system packages not available"
    
    # Image processing libraries
    print_status "Installing image processing libraries..."
    sudo apt install -y \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        || print_warning "Some image libraries may have failed"
    
    # Optimized math libraries for Pi
    print_status "Installing optimized math libraries..."
    sudo apt install -y \
        libopenblas-dev \
        libatlas-base-dev \
        gfortran \
        || print_warning "Some math libraries may have failed"
    
    # Raspberry Pi specific packages
    if check_raspberry_pi; then
        print_status "Installing Raspberry Pi specific packages..."
        sudo apt install -y \
            python3-picamera2 \
            python3-rpi.gpio \
            libraspberrypi-bin \
            || print_warning "Some Pi-specific packages may have failed"
        
        # Enable camera interface
        print_status "Enabling camera interface..."
        sudo raspi-config nonint do_camera 0 || print_warning "Could not enable camera automatically"
    fi
    
    print_success "System dependencies installed"
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip (with memory limits)
    if [[ "$LOW_MEMORY" == true ]]; then
        pip install --upgrade --no-cache-dir pip setuptools wheel
    else
        pip install --upgrade pip setuptools wheel
    fi
    
    print_success "Virtual environment created"
}

# Install Python dependencies with memory optimization
install_python_deps() {
    print_status "Installing Python dependencies (this may take 30+ minutes on Pi)..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Set memory-optimized build flags
    export PIP_NO_CACHE_DIR=1
    export MAX_JOBS=1
    
    if [[ "$LOW_MEMORY" == true ]]; then
        print_warning "Using low memory mode - installation will be slower"
        export CFLAGS="-O2 -march=native -mtune=native"
    fi
    
    # Install cmake (Python package)
    print_status "Installing cmake..."
    pip install --no-cache-dir cmake || print_warning "cmake install failed"
    
    # Install dlib (takes the longest on Pi)
    print_status "Installing dlib (this will take 15-30 minutes)..."
    print_status "Please be patient, do not interrupt..."
    
    pip install --no-cache-dir --verbose dlib || {
        print_warning "dlib pip install failed, trying alternative method..."
        # Try with specific optimizations for Pi
        CFLAGS="-O2 -march=native" pip install --no-cache-dir --verbose dlib
    }
    
    # Install face_recognition
    print_status "Installing face_recognition..."
    pip install --no-cache-dir face_recognition || {
        print_error "face_recognition installation failed"
        print_status "You may need to install manually later"
    }
    
    # Install OpenCV (use headless for Pi)
    print_status "Installing OpenCV (headless version for Pi)..."
    pip install --no-cache-dir opencv-python-headless || {
        print_warning "OpenCV headless failed, trying standard version..."
        pip install --no-cache-dir opencv-python
    }
    
    # Install project dependencies
    print_status "Installing project dependencies..."
    
    if [ -f "requirements-pi.txt" ]; then
        pip install --no-cache-dir -r requirements-pi.txt || print_warning "Some Pi requirements may have failed"
    elif [ -f "requirements.txt" ]; then
        pip install --no-cache-dir -r requirements.txt || print_warning "Some requirements may have failed"
    fi
    
    # Install optional monitoring tools
    if [ -f "pyproject.toml" ]; then
        pip install --no-cache-dir -e ".[pi,monitoring]" || print_warning "Some optional dependencies may have failed"
    fi
    
    print_success "Python dependencies installed"
}

# Setup project directories
setup_directories() {
    print_status "Setting up project directories..."
    
    # Create data directories
    mkdir -p data/{known_faces,blacklist_faces,captures,logs}
    mkdir -p config
    
    # Create empty __init__.py files if they don't exist
    touch src/__init__.py 2>/dev/null || true
    touch config/__init__.py 2>/dev/null || true
    
    # Set appropriate permissions
    chmod -R u+rwX,go+rX data/
    
    print_success "Project directories created"
}

# Setup configuration for Raspberry Pi
setup_config() {
    print_status "Setting up Raspberry Pi configuration..."
    
    # Copy credentials template if not exists
    if [ ! -f "config/credentials_telegram.py" ] && [ -f "config/credentials_template.py" ]; then
        cp config/credentials_template.py config/credentials_telegram.py
        print_warning "Telegram credentials template copied to config/credentials_telegram.py"
        print_warning "Please edit this file with your actual bot token and chat ID"
    fi
    
    # Create environment file for Pi
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Doorbell Security System Environment Variables (Raspberry Pi)

# Platform detection
PLATFORM_TYPE=raspberry_pi

# GPIO Configuration
DOORBELL_PIN=18
RED_LED_PIN=16
YELLOW_LED_PIN=20
GREEN_LED_PIN=21

# Camera Configuration
CAMERA_RESOLUTION=1280,720
CAMERA_ROTATION=0
CAMERA_BRIGHTNESS=50
CAMERA_CONTRAST=0

# Face Recognition (optimized for Pi)
FACE_TOLERANCE=0.6
BLACKLIST_TOLERANCE=0.5
REQUIRE_FACE_FOR_CAPTURE=True
FACE_DETECTION_MODEL=hog

# Performance (optimized for Pi)
WORKER_PROCESSES=1
MEMORY_LIMIT_MB=1024
LOW_MEMORY_MODE=true

# Notifications
NOTIFY_ON_KNOWN_VISITORS=False
INCLUDE_PHOTO_FOR_KNOWN=False

# System
LOG_LEVEL=INFO
EOF
        print_success "Raspberry Pi environment configuration created"
    else
        print_status "Environment file already exists, skipping..."
    fi
}

# Setup systemd service
setup_service() {
    print_status "Setting up systemd service..."
    
    INSTALL_DIR=$(pwd)
    USER=$(whoami)
    
    # Create service file
    sudo tee /etc/systemd/system/doorbell-security.service > /dev/null << EOF
[Unit]
Description=Doorbell Face Recognition Security System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/app.py
Restart=always
RestartSec=10
StandardOutput=append:$INSTALL_DIR/data/logs/doorbell.log
StandardError=append:$INSTALL_DIR/data/logs/doorbell.err

# Resource limits for Pi
MemoryLimit=1G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable doorbell-security.service
    
    print_success "Systemd service configured"
    print_status "Service commands:"
    echo "  Start:   sudo systemctl start doorbell-security"
    echo "  Stop:    sudo systemctl stop doorbell-security"
    echo "  Status:  sudo systemctl status doorbell-security"
    echo "  Logs:    sudo journalctl -u doorbell-security -f"
}

# Create management scripts
create_scripts() {
    print_status "Creating management scripts..."
    
    mkdir -p scripts
    
    # Test script
    cat > scripts/test-pi.sh << 'EOF'
#!/bin/bash
echo "🍓 Testing Doorbell Security System on Raspberry Pi..."
source venv/bin/activate

# Run platform detection test
python3 -c "
import sys
sys.path.append('.')

from src.platform_detector import PlatformDetector
from config.platform_configs import get_platform_configs

print('🍓 Testing Raspberry Pi platform detection...')
print()

detector = PlatformDetector()
info = detector.get_platform_info()

print('Platform Information:')
print(f'  Model: Raspberry Pi')
print(f'  Architecture: {info[\"architecture\"]}')
print(f'  Python: {info[\"python_version\"]}')
print(f'  Memory: {info[\"memory_gb\"]}GB')
print(f'  CPU Count: {info[\"cpu_count\"]}')
print(f'  Has GPIO: {info[\"has_gpio\"]}')
print(f'  Has Camera: {info[\"has_camera\"]}')
print()

print('Platform Configuration:')
config = get_platform_configs(detector)
print(f'  Platform Type: {config.get(\"platform_type\")}')
print(f'  Worker Processes: {config.get(\"worker_processes\")}')
print(f'  Memory Limit: {config.get(\"memory_limit_mb\")}MB')
print(f'  Low Memory Mode: {config.get(\"low_memory_mode\")}')
print(f'  Face Detection Model: {config.get(\"face_detection_model\")}')
print()

print('✅ Platform detection test completed!')
"

# Test imports
python3 -c "
print('Testing imports...')
try:
    import face_recognition
    print('✅ face_recognition: OK')
except ImportError as e:
    print(f'❌ face_recognition: {e}')

try:
    import cv2
    print('✅ opencv: OK')
except ImportError as e:
    print(f'❌ opencv: {e}')

try:
    import RPi.GPIO as GPIO
    print('✅ RPi.GPIO: OK')
except ImportError:
    print('⚠️  RPi.GPIO: NOT AVAILABLE (may need picamera2)')

print()
print('🎉 Raspberry Pi test completed!')
print('💡 To start as service: sudo systemctl start doorbell-security')
"
EOF
    chmod +x scripts/test-pi.sh
    
    print_success "Management scripts created"
}

# Run validation
validate_installation() {
    print_status "Validating installation..."
    
    source venv/bin/activate
    
    # Test Python version
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_status "Python version: $python_version"
    
    # Test critical imports
    python -c "import face_recognition" 2>/dev/null && print_success "face_recognition: OK" || print_warning "face_recognition: NOT INSTALLED"
    python -c "import cv2" 2>/dev/null && print_success "opencv: OK" || print_warning "opencv: NOT INSTALLED"
    python -c "import numpy" 2>/dev/null && print_success "numpy: OK" || print_warning "numpy: NOT INSTALLED"
    
    print_success "Validation complete"
}

# Main setup function
main() {
    echo ""
    print_status "Starting Doorbell Security System setup for Raspberry Pi..."
    echo ""
    
    # Check if script is run from correct directory
    if [ ! -f "pyproject.toml" ] && [ ! -f "app.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_raspberry_pi
    check_memory
    configure_swap
    update_system
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    setup_config
    setup_service
    create_scripts
    validate_installation
    
    echo ""
    print_success "Raspberry Pi setup completed successfully! 🎉"
    echo ""
    print_status "Next steps:"
    echo "1. Edit config/credentials_telegram.py with your Telegram bot credentials"
    echo "2. Add known face images to data/known_faces/ directory"
    echo "3. Test the system: ./scripts/test-pi.sh"
    echo "4. Start the service: sudo systemctl start doorbell-security"
    echo "5. Check status: sudo systemctl status doorbell-security"
    echo ""
    print_status "🍓 Raspberry Pi Optimization Notes:"
    echo "   - Low memory mode enabled"
    echo "   - HOG face detection model (CPU efficient)"
    echo "   - Single worker process"
    echo "   - Resource limits configured in systemd"
    echo ""
    print_status "📝 For help and documentation, see README.md"
    echo ""
}

# Run main function
main "$@"
