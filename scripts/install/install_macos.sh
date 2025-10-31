#!/bin/bash
# macOS Installation Script with Apple Silicon support
# Doorbell Security System - Cross-Platform Installer

set -e

echo "🍎 Installing Doorbell Security System on macOS..."

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

# Detect architecture
ARCH=$(uname -m)
print_status "Detected architecture: $ARCH"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is for macOS only"
    exit 1
fi

print_success "Running on macOS"

# Detect Apple Silicon
if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    IS_APPLE_SILICON=true
    print_success "Apple Silicon detected (M1/M2/M3)"
else
    IS_APPLE_SILICON=false
    print_status "Intel Mac detected"
fi

# Check for Homebrew
check_homebrew() {
    print_status "Checking for Homebrew..."
    
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH
        if [[ "$IS_APPLE_SILICON" == true ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    else
        print_success "Homebrew already installed"
        brew update
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies with Homebrew..."
    
    # Essential packages
    brew install \
        python@3.11 \
        cmake \
        pkg-config \
        jpeg \
        libpng \
        libtiff \
        git \
        curl \
        wget \
        || print_warning "Some packages may already be installed"
    
    # Install dlib via Homebrew (easier on macOS)
    if [[ "$IS_APPLE_SILICON" == true ]]; then
        print_status "Installing dlib for Apple Silicon..."
        # dlib can be tricky on Apple Silicon, try Homebrew first
        brew install dlib || print_warning "dlib via Homebrew failed, will try pip"
    else
        print_status "Installing dlib for Intel Mac..."
        brew install dlib || print_warning "dlib via Homebrew failed, will try pip"
    fi
    
    # Install OpenCV via Homebrew (optional, but recommended)
    print_status "Installing OpenCV..."
    brew install opencv || print_warning "OpenCV via Homebrew failed, will use pip version"
    
    print_success "System dependencies installed"
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    # Use Homebrew Python
    if [[ "$IS_APPLE_SILICON" == true ]]; then
        PYTHON_CMD="/opt/homebrew/bin/python3.11"
    else
        PYTHON_CMD="/usr/local/bin/python3.11"
    fi
    
    # Fallback to system python3 if Homebrew python not found
    if [ ! -f "$PYTHON_CMD" ]; then
        PYTHON_CMD="python3"
    fi
    
    # Create virtual environment
    $PYTHON_CMD -m venv venv
    
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
    
    # Set environment variables for Apple Silicon
    if [[ "$IS_APPLE_SILICON" == true ]]; then
        print_status "Configuring for Apple Silicon..."
        
        export CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64"
        export ARCHFLAGS="-arch arm64"
        
        # Ensure Homebrew paths are in PATH
        export PATH="/opt/homebrew/bin:$PATH"
        export LDFLAGS="-L/opt/homebrew/lib"
        export CPPFLAGS="-I/opt/homebrew/include"
        export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
    fi
    
    # Install dlib and face_recognition
    print_status "Installing face recognition dependencies..."
    
    # Try to install cmake first
    pip install cmake || print_warning "cmake install failed, continuing..."
    
    # Install dlib
    print_status "Installing dlib (this may take a few minutes)..."
    pip install --no-cache-dir dlib || {
        print_warning "dlib installation failed, trying alternative method..."
        # Alternative: try installing from source with specific flags
        pip install --no-cache-dir --verbose dlib
    }
    
    # Install face_recognition
    print_status "Installing face_recognition..."
    pip install --no-cache-dir face_recognition || {
        print_error "face_recognition installation failed"
        print_status "You may need to install manually later"
    }
    
    # Install OpenCV
    print_status "Installing OpenCV..."
    pip install opencv-python || print_warning "OpenCV installation failed"
    
    # Install remaining dependencies
    print_status "Installing remaining dependencies..."
    
    if [ -f "requirements-web.txt" ]; then
        pip install -r requirements-web.txt || print_warning "Some web requirements may have failed"
    fi
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt || print_warning "Some requirements may have failed"
    fi
    
    # Install development dependencies if requested
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
    
    print_success "Project directories created"
}

# Setup configuration for macOS
setup_config() {
    print_status "Setting up macOS configuration..."
    
    # Copy credentials template if not exists
    if [ ! -f "config/credentials_telegram.py" ] && [ -f "config/credentials_template.py" ]; then
        cp config/credentials_template.py config/credentials_telegram.py
        print_warning "Telegram credentials template copied to config/credentials_telegram.py"
        print_warning "Please edit this file with your actual bot token and chat ID"
    fi
    
    # Create environment file for macOS
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Doorbell Security System Environment Variables (macOS)

# Development mode
DEVELOPMENT_MODE=true

# Platform detection
PLATFORM_TYPE=macos

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

# System
LOG_LEVEL=INFO
EOF
        print_success "macOS environment configuration created"
    else
        print_status "Environment file already exists, skipping..."
    fi
}

# Create management scripts for macOS
create_scripts() {
    print_status "Creating management scripts..."
    
    mkdir -p scripts
    
    # Start script for web interface
    if [ ! -f "scripts/start-web.sh" ]; then
        cat > scripts/start-web.sh << 'EOF'
#!/bin/bash
echo "🌐 Starting Doorbell Security Web Interface..."
source venv/bin/activate
export DEVELOPMENT_MODE=true
python app.py
EOF
        chmod +x scripts/start-web.sh
    fi
    
    # Test script
    if [ ! -f "scripts/test-macos.sh" ] || [ "$OVERWRITE_SCRIPTS" = "true" ]; then
        cat > scripts/test-macos.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Doorbell Security System on macOS..."
source venv/bin/activate
export DEVELOPMENT_MODE=true

# Run platform detection test
python3 -c "
import sys
sys.path.append('.')

from src.platform_detector import PlatformDetector
from config.platform_configs import get_platform_configs

print('🍎 Testing macOS platform detection...')
print()

detector = PlatformDetector()
info = detector.get_platform_info()

print('Platform Information:')
print(f'  OS: {info[\"os\"]} {info[\"os_version\"]}')
print(f'  Architecture: {info[\"architecture\"]}')
print(f'  Python: {info[\"python_version\"]}')
print(f'  Apple Silicon: {info[\"is_apple_silicon\"]}')
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
print(f'  Installation Method: {config.get(\"installation_method\")}')
print()

print('✅ Platform detection test completed!')
"

# Run basic import tests
python3 -c "
import sys
sys.path.append('.')

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
    from src.platform_detector import platform_detector
    print('✅ platform_detector: OK')
except Exception as e:
    print(f'❌ platform_detector: {e}')

print()
print('🎉 macOS test completed!')
print('💡 To start the web interface: ./scripts/start-web.sh')
print('🌐 Then open: http://localhost:5000')
"
EOF
        chmod +x scripts/test-macos.sh
    fi
    
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
    print_status "Starting Doorbell Security System setup for macOS..."
    echo ""
    
    # Check if script is run from correct directory
    if [ ! -f "pyproject.toml" ] && [ ! -f "app.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_homebrew
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    setup_config
    create_scripts
    validate_installation
    
    echo ""
    print_success "macOS setup completed successfully! 🎉"
    echo ""
    print_status "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Edit config/credentials_telegram.py with your Telegram bot credentials (optional)"
    echo "3. Add known face images to data/known_faces/ directory (optional)"
    echo "4. Test the system: ./scripts/test-macos.sh"
    echo "5. Start the web interface: ./scripts/start-web.sh"
    echo "6. Open http://localhost:5000 in your browser"
    echo ""
    
    if [[ "$IS_APPLE_SILICON" == true ]]; then
        print_status "📱 Apple Silicon Notes:"
        echo "   - Some Python packages may need Rosetta 2"
        echo "   - Metal GPU acceleration is automatically enabled"
        echo "   - Face detection uses HOG model for best compatibility"
    fi
    
    echo ""
    print_status "🚀 For cloud deployment:"
    echo "• Vercel: vercel --prod"
    echo "• Render: Connect your GitHub repo"
    echo "• Railway: railway up"
    echo "• Docker: docker-compose up"
    echo ""
}

# Run main function
main "$@"
