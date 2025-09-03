#!/bin/bash
# Doorbell Security System Setup Script

set -e  # Exit on any error

echo "🔒 Doorbell Face Recognition Security System Setup"
echo "=================================================="

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
    return 1
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    print_success "System updated"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Essential packages
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        cmake \
        pkg-config \
        libopencv-dev \
        libatlas-base-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        git \
        curl \
        wget
    
    # Raspberry Pi specific packages
    if check_raspberry_pi; then
        print_status "Installing Raspberry Pi specific packages..."
        sudo apt install -y \
            python3-picamera2 \
            python3-rpi.gpio \
            libraspberrypi-bin
        
        # Enable camera interface
        print_status "Enabling camera interface..."
        sudo raspi-config nonint do_camera 0
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
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dlib with optimizations for Raspberry Pi
    if check_raspberry_pi; then
        print_status "Installing dlib (optimized for Raspberry Pi)..."
        pip install dlib==19.24.2 --verbose
    else
        pip install dlib
    fi
    
    # Install other requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Setup project directories
setup_directories() {
    print_status "Setting up project directories..."
    
    # Create data directories
    mkdir -p data/{known_faces,blacklist_faces,captures,logs}
    
    # Create empty __init__.py files
    touch src/__init__.py
    touch config/__init__.py
    
    # Set permissions
    chmod +x scripts/*.sh 2>/dev/null || true
    
    print_success "Project directories created"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    # Copy credentials template if not exists
    if [ ! -f "config/credentials_telegram.py" ]; then
        cp config/credentials_template.py config/credentials_telegram.py
        print_warning "Telegram credentials template copied to config/credentials_telegram.py"
        print_warning "Please edit this file with your actual bot token and chat ID"
    fi
    
    # Create environment file
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Doorbell Security System Environment Variables

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

# Face Recognition
FACE_TOLERANCE=0.6
BLACKLIST_TOLERANCE=0.5
REQUIRE_FACE_FOR_CAPTURE=True

# Notifications
NOTIFY_ON_KNOWN_VISITORS=False
INCLUDE_PHOTO_FOR_KNOWN=False

# FBI Integration
FBI_UPDATE_ENABLED=True
FBI_MAX_ENTRIES=50

# System
LOG_LEVEL=INFO
EOF
        print_success "Environment configuration created"
    fi
}

# Setup systemd service
setup_service() {
    print_status "Setting up systemd service..."
    
    INSTALL_DIR=$(pwd)
    
    # Create service file
    sudo tee /etc/systemd/system/doorbell-security.service > /dev/null << EOF
[Unit]
Description=Doorbell Face Recognition Security System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/src/doorbell_security.py
Restart=always
RestartSec=5
StandardOutput=append:$INSTALL_DIR/data/logs/doorbell.log
StandardError=append:$INSTALL_DIR/data/logs/doorbell.err

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable doorbell-security.service
    
    print_success "Systemd service configured"
}

# Create management scripts
create_scripts() {
    print_status "Creating management scripts..."
    
    mkdir -p scripts
    
    # Start script
    cat > scripts/start.sh << 'EOF'
#!/bin/bash
echo "Starting Doorbell Security System..."
sudo systemctl start doorbell-security.service
sudo systemctl status doorbell-security.service
EOF
    
    # Stop script
    cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "Stopping Doorbell Security System..."
sudo systemctl stop doorbell-security.service
EOF
    
    # Status script
    cat > scripts/status.sh << 'EOF'
#!/bin/bash
echo "Doorbell Security System Status:"
echo "================================"
sudo systemctl status doorbell-security.service
echo ""
echo "Recent logs:"
echo "============"
tail -n 20 data/logs/doorbell.log
EOF
    
    # Test script
    cat > scripts/test.sh << 'EOF'
#!/bin/bash
echo "Testing Doorbell Security System..."
source venv/bin/activate
python3 -c "
from src.camera_handler import CameraHandler
from src.telegram_notifier import TelegramNotifier
from src.gpio_handler import GPIOHandler
from src.face_manager import FaceManager

print('Testing components...')

# Test camera
try:
    camera = CameraHandler()
    camera.initialize()
    print('✅ Camera: OK')
    camera.cleanup()
except Exception as e:
    print(f'❌ Camera: {e}')

# Test Telegram
try:
    notifier = TelegramNotifier()
    if notifier.test_notification():
        print('✅ Telegram: OK')
    else:
        print('❌ Telegram: Not configured')
except Exception as e:
    print(f'❌ Telegram: {e}')

# Test GPIO
try:
    gpio = GPIOHandler()
    gpio.test_leds()
    print('✅ GPIO: OK')
    gpio.cleanup()
except Exception as e:
    print(f'❌ GPIO: {e}')

# Test Face Recognition
try:
    face_manager = FaceManager()
    face_manager.load_known_faces()
    face_manager.load_blacklist_faces()
    stats = face_manager.get_stats()
    print(f'✅ Face Recognition: {stats}')
except Exception as e:
    print(f'❌ Face Recognition: {e}')

print('Test completed!')
"
EOF
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_success "Management scripts created"
}

# Main setup function
main() {
    echo ""
    print_status "Starting Doorbell Security System setup..."
    echo ""
    
    # Check if script is run from correct directory
    if [ ! -f "requirements.txt" ] || [ ! -f "src/doorbell_security.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_raspberry_pi
    update_system
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    setup_config
    setup_service
    create_scripts
    
    echo ""
    print_success "Setup completed successfully! 🎉"
    echo ""
    print_status "Next steps:"
    echo "1. Edit config/credentials_telegram.py with your Telegram bot credentials"
    echo "2. Add known face images to data/known_faces/ directory"
    echo "3. Test the system: ./scripts/test.sh"
    echo "4. Start the system: ./scripts/start.sh"
    echo ""
    print_status "For help and documentation, see README.md"
}

# Run main function
main "$@"
