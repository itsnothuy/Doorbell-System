#!/bin/bash
# Doorbell Security System Setup Script for macOS

set -e  # Exit on any error

echo "🍎 Doorbell Security System - macOS Setup"
echo "========================================"

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

# Check if running on macOS
check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is for macOS only"
        exit 1
    fi
    
    print_success "Running on macOS"
}

# Check for Homebrew
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        print_success "Homebrew already installed"
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies with Homebrew..."
    
    # Install essential packages
    brew install \
        python@3.11 \
        cmake \
        pkg-config \
        opencv \
        jpeg \
        libpng \
        libtiff \
        git \
        curl \
        wget
    
    print_success "System dependencies installed"
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    # Use Homebrew Python
    /opt/homebrew/bin/python3.11 -m venv venv || python3 -m venv venv
    
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
    
    # Install requirements for web deployment (lighter)
    pip install -r requirements-web.txt
    
    # Add Flask for web interface
    pip install flask flask-cors gunicorn
    
    print_success "Python dependencies installed"
}

# Setup project directories
setup_directories() {
    print_status "Setting up project directories..."
    
    # Create data directories
    mkdir -p data/{known_faces,blacklist_faces,captures,logs}
    mkdir -p templates static
    
    # Create empty __init__.py files
    touch src/__init__.py
    touch config/__init__.py
    
    print_success "Project directories created"
}

# Setup configuration for macOS
setup_config() {
    print_status "Setting up macOS configuration..."
    
    # Copy credentials template if not exists
    if [ ! -f "config/credentials_telegram.py" ]; then
        cp config/credentials_template.py config/credentials_telegram.py
        print_warning "Telegram credentials template copied to config/credentials_telegram.py"
        print_warning "Please edit this file with your actual bot token and chat ID"
    fi
    
    # Create environment file for macOS
    cat > .env << 'EOF'
# Doorbell Security System Environment Variables (macOS)

# Development mode
DEVELOPMENT_MODE=true

# Web interface
PORT=5000

# Camera Configuration (webcam)
CAMERA_RESOLUTION=1280,720
CAMERA_ROTATION=0
CAMERA_BRIGHTNESS=50
CAMERA_CONTRAST=0

# Face Recognition
FACE_TOLERANCE=0.6
BLACKLIST_TOLERANCE=0.5
REQUIRE_FACE_FOR_CAPTURE=False

# Notifications
NOTIFY_ON_KNOWN_VISITORS=False
INCLUDE_PHOTO_FOR_KNOWN=False

# FBI Integration
FBI_UPDATE_ENABLED=True
FBI_MAX_ENTRIES=50

# System
LOG_LEVEL=INFO
EOF
    
    print_success "macOS configuration created"
}

# Create management scripts for macOS
create_scripts() {
    print_status "Creating management scripts..."
    
    mkdir -p scripts
    
    # Start script for web interface
    cat > scripts/start-web.sh << 'EOF'
#!/bin/bash
echo "🌐 Starting Doorbell Security Web Interface..."
source venv/bin/activate
export DEVELOPMENT_MODE=true
python app.py
EOF
    
    # Test script
    cat > scripts/test-macos.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Doorbell Security System on macOS..."
source venv/bin/activate
export DEVELOPMENT_MODE=true

python3 -c "
import sys
sys.path.append('.')
from src.platform_detector import platform_detector
from src.camera_handler import CameraHandler
from src.telegram_notifier import TelegramNotifier
from src.gpio_handler import GPIOHandler
from src.face_manager import FaceManager

print('🍎 Testing macOS components...')
print(f'Platform: {platform_detector.system} {platform_detector.machine}')
print(f'Development mode: {platform_detector.is_development}')
print()

# Test camera
try:
    camera = CameraHandler()
    camera.initialize()
    print('✅ Camera: OK (Mock mode for testing)')
    camera.cleanup()
except Exception as e:
    print(f'❌ Camera: {e}')

# Test Telegram
try:
    notifier = TelegramNotifier()
    if notifier.initialized:
        print('✅ Telegram: OK')
    else:
        print('⚠️  Telegram: Not configured (edit credentials_telegram.py)')
except Exception as e:
    print(f'❌ Telegram: {e}')

# Test GPIO (mock)
try:
    gpio = GPIOHandler()
    print('✅ GPIO: OK (Mock mode for testing)')
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

print()
print('🎉 macOS test completed!')
print('💡 To start the web interface: ./scripts/start-web.sh')
print('🌐 Then open: http://localhost:5000')
"
EOF
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_success "Management scripts created"
}

# Main setup function
main() {
    echo ""
    print_status "Starting Doorbell Security System setup for macOS..."
    echo ""
    
    # Check if script is run from correct directory
    if [ ! -f "requirements-web.txt" ] || [ ! -f "app.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_macos
    check_homebrew
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    setup_config
    create_scripts
    
    echo ""
    print_success "macOS setup completed successfully! 🎉"
    echo ""
    print_status "Next steps:"
    echo "1. Edit config/credentials_telegram.py with your Telegram bot credentials (optional)"
    echo "2. Add known face images to data/known_faces/ directory (optional)"
    echo "3. Test the system: ./scripts/test-macos.sh"
    echo "4. Start the web interface: ./scripts/start-web.sh"
    echo "5. Open http://localhost:5000 in your browser"
    echo ""
    print_status "📱 Use the web interface to simulate doorbell presses and test all features!"
    echo ""
    print_status "🚀 For cloud deployment:"
    echo "• Vercel: vercel --prod"
    echo "• Render: Connect your GitHub repo"
    echo "• Railway: railway up"
    echo "• Docker: docker-compose up"
}

# Run main function
main "$@"
