#!/bin/bash
# Ubuntu/Debian Installation Script
# Doorbell Security System - Cross-Platform Installer

set -e

echo "🐧 Installing Doorbell Security System on Ubuntu/Debian..."

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

# Check if running on Ubuntu/Debian
check_system() {
    print_status "Checking system..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
        
        if [[ "$OS" != "ubuntu" ]] && [[ "$OS" != "debian" ]] && [[ "$OS" != "raspbian" ]]; then
            print_warning "This script is optimized for Ubuntu/Debian"
            print_status "Detected: $OS $VERSION"
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_success "Detected: $OS $VERSION"
        fi
    else
        print_warning "Could not detect OS version"
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
    print_status "Installing system dependencies..."
    
    # Essential build tools
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
    
    # Image processing libraries
    print_status "Installing image processing libraries..."
    sudo apt install -y \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        || print_warning "Some image libraries may have failed"
    
    # Linear algebra and optimization libraries
    print_status "Installing math libraries..."
    sudo apt install -y \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        || print_warning "Some math libraries may have failed"
    
    # dlib dependencies
    print_status "Installing dlib dependencies..."
    sudo apt install -y \
        libboost-all-dev \
        libdlib-dev \
        || print_warning "dlib development libraries not available via apt, will build from source"
    
    # Optional: Try to install Python packages from apt (faster if available)
    print_status "Attempting to install Python packages from apt (optional)..."
    sudo apt install -y \
        python3-numpy \
        python3-opencv \
        python3-pil \
        python3-requests \
        python3-yaml \
        || print_status "System Python packages not all available, will use pip"
    
    # Optional: GTK for OpenCV GUI (if needed)
    sudo apt install -y \
        libgtk-3-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        || print_warning "GTK libraries not installed (not required for headless)"
    
    print_success "System dependencies installed"
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install cmake (Python package)
    print_status "Installing cmake Python package..."
    pip install cmake || print_warning "cmake install failed"
    
    # Install dlib
    print_status "Installing dlib (this may take several minutes)..."
    pip install --no-cache-dir dlib || {
        print_warning "dlib pip install failed, trying from source..."
        # If dlib install fails, try building from source with verbose output
        pip install --no-cache-dir --verbose dlib
    }
    
    # Install face_recognition
    print_status "Installing face_recognition..."
    pip install --no-cache-dir face_recognition || {
        print_error "face_recognition installation failed"
        print_status "You may need to check dependencies"
    }
    
    # Install OpenCV
    print_status "Installing OpenCV..."
    pip install opencv-python || print_warning "OpenCV installation failed"
    
    # Install project dependencies
    print_status "Installing project dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt || print_warning "Some requirements may have failed"
    fi
    
    if [ -f "requirements-web.txt" ]; then
        pip install -r requirements-web.txt || print_warning "Some web requirements may have failed"
    fi
    
    # Install development dependencies if available
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev,monitoring]" || print_warning "Some optional dependencies may have failed"
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

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    # Copy credentials template if not exists
    if [ ! -f "config/credentials_telegram.py" ] && [ -f "config/credentials_template.py" ]; then
        cp config/credentials_template.py config/credentials_telegram.py
        print_warning "Telegram credentials template copied to config/credentials_telegram.py"
        print_warning "Please edit this file with your actual bot token and chat ID"
    fi
    
    # Create environment file
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Doorbell Security System Environment Variables (Ubuntu/Debian)

# Development mode
DEVELOPMENT_MODE=false

# Platform detection
PLATFORM_TYPE=ubuntu

# Web interface
PORT=5000

# Camera Configuration
CAMERA_RESOLUTION=1280,720
CAMERA_ROTATION=0
CAMERA_BRIGHTNESS=50
CAMERA_CONTRAST=0

# Face Recognition
FACE_TOLERANCE=0.6
BLACKLIST_TOLERANCE=0.5
REQUIRE_FACE_FOR_CAPTURE=True

# Notifications
NOTIFY_ON_KNOWN_VISITORS=False
INCLUDE_PHOTO_FOR_KNOWN=False

# System
LOG_LEVEL=INFO
EOF
        print_success "Environment configuration created"
    else
        print_status "Environment file already exists, skipping..."
    fi
}

# Create management scripts
create_scripts() {
    print_status "Creating management scripts..."
    
    mkdir -p scripts
    
    # Start script
    cat > scripts/start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Doorbell Security System..."
source venv/bin/activate
python app.py
EOF
    chmod +x scripts/start.sh
    
    # Test script
    cat > scripts/test-ubuntu.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Doorbell Security System on Ubuntu..."
source venv/bin/activate

# Run platform detection test
python3 -c "
import sys
sys.path.append('.')

from src.platform_detector import PlatformDetector
from config.platform_configs import get_platform_configs

print('🐧 Testing Ubuntu platform detection...')
print()

detector = PlatformDetector()
info = detector.get_platform_info()

print('Platform Information:')
print(f'  OS: {info[\"os\"]} {info[\"os_version\"]}')
print(f'  Architecture: {info[\"architecture\"]}')
print(f'  Python: {info[\"python_version\"]}')
print(f'  Memory: {info[\"memory_gb\"]}GB')
print(f'  CPU Count: {info[\"cpu_count\"]}')
print(f'  Has GPU: {info[\"has_gpu\"]}')
print(f'  Has Camera: {info[\"has_camera\"]}')
print()

print('Platform Configuration:')
config = get_platform_configs(detector)
print(f'  Platform Type: {config.get(\"platform_type\")}')
print(f'  Worker Processes: {config.get(\"worker_processes\")}')
print(f'  Memory Limit: {config.get(\"memory_limit_mb\")}MB')
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

print()
print('🎉 Ubuntu test completed!')
print('💡 To start the application: ./scripts/start.sh')
"
EOF
    chmod +x scripts/test-ubuntu.sh
    
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
    print_status "Starting Doorbell Security System setup for Ubuntu/Debian..."
    echo ""
    
    # Check if script is run from correct directory
    if [ ! -f "pyproject.toml" ] && [ ! -f "app.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root - this is not recommended"
        print_status "Some operations will be performed without sudo"
    fi
    
    # Run setup steps
    check_system
    update_system
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    setup_config
    create_scripts
    validate_installation
    
    echo ""
    print_success "Ubuntu/Debian setup completed successfully! 🎉"
    echo ""
    print_status "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Edit config/credentials_telegram.py with your Telegram bot credentials (optional)"
    echo "3. Add known face images to data/known_faces/ directory"
    echo "4. Test the system: ./scripts/test-ubuntu.sh"
    echo "5. Start the application: ./scripts/start.sh"
    echo "6. Access web interface: http://localhost:5000"
    echo ""
    print_status "📝 For systemd service setup, see documentation"
    print_status "🐳 For Docker deployment: docker-compose up"
    echo ""
}

# Run main function
main "$@"
