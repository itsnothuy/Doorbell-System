#!/bin/bash
# Universal Development Environment Setup Script
# Doorbell Security System - Cross-Platform Installer

set -e

echo "🔧 Doorbell Security System - Universal Setup"
echo "=============================================="
echo ""

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

# Detect platform
detect_platform() {
    print_status "Detecting platform..."
    
    OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
    
    case "$OS_TYPE" in
        darwin*)
            PLATFORM="macos"
            if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
                PLATFORM_VARIANT="apple_silicon"
                print_success "Detected: macOS on Apple Silicon"
            else
                PLATFORM_VARIANT="intel"
                print_success "Detected: macOS on Intel"
            fi
            ;;
        linux*)
            # Check if Raspberry Pi
            if [ -f /proc/device-tree/model ]; then
                PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null)
                if [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
                    PLATFORM="raspberry_pi"
                    PLATFORM_VARIANT="pi"
                    print_success "Detected: $PI_MODEL"
                else
                    PLATFORM="linux"
                    PLATFORM_VARIANT="ubuntu"
                    print_success "Detected: Linux"
                fi
            else
                PLATFORM="linux"
                PLATFORM_VARIANT="ubuntu"
                print_success "Detected: Linux"
            fi
            ;;
        mingw*|msys*|cygwin*)
            PLATFORM="windows"
            PLATFORM_VARIANT="windows"
            print_success "Detected: Windows (via $OS_TYPE)"
            ;;
        *)
            PLATFORM="unknown"
            PLATFORM_VARIANT="unknown"
            print_error "Unsupported platform: $OS_TYPE"
            exit 1
            ;;
    esac
    
    print_status "Platform: $PLATFORM ($PLATFORM_VARIANT)"
    print_status "Architecture: $ARCH"
}

# Run platform-specific installer
run_platform_installer() {
    print_status "Running platform-specific installer..."
    echo ""
    
    case "$PLATFORM" in
        macos)
            if [ -f "scripts/install/install_macos.sh" ]; then
                bash scripts/install/install_macos.sh
            else
                print_error "macOS installer not found"
                exit 1
            fi
            ;;
        raspberry_pi)
            if [ -f "scripts/install/install_raspberry_pi.sh" ]; then
                bash scripts/install/install_raspberry_pi.sh
            else
                print_error "Raspberry Pi installer not found"
                exit 1
            fi
            ;;
        linux)
            if [ -f "scripts/install/install_ubuntu.sh" ]; then
                bash scripts/install/install_ubuntu.sh
            else
                print_error "Ubuntu installer not found"
                exit 1
            fi
            ;;
        windows)
            print_error "Please run scripts/install/install_windows.ps1 instead"
            print_status "Open PowerShell and run:"
            print_status "  Set-ExecutionPolicy Bypass -Scope Process -Force"
            print_status "  .\\scripts\\install\\install_windows.ps1"
            exit 1
            ;;
        *)
            print_error "Unsupported platform: $PLATFORM"
            exit 1
            ;;
    esac
}

# Setup pre-commit hooks
setup_precommit() {
    if command -v pre-commit &> /dev/null; then
        print_status "Setting up pre-commit hooks..."
        source venv/bin/activate 2>/dev/null || true
        pre-commit install || print_warning "Could not install pre-commit hooks"
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not available"
        print_status "To install: pip install pre-commit && pre-commit install"
    fi
}

# Create platform-specific configuration
create_platform_config() {
    print_status "Creating platform-specific configuration..."
    
    source venv/bin/activate 2>/dev/null || true
    
    python3 << 'EOF' || print_warning "Could not create platform config"
import sys
import json
from pathlib import Path

sys.path.append('.')

try:
    from src.platform_detector import PlatformDetector
    from config.platform_configs import get_platform_configs
    
    detector = PlatformDetector()
    config = get_platform_configs(detector)
    
    # Save configuration
    config_file = Path('config/platform_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f'✓ Platform configuration saved to {config_file}')
except Exception as e:
    print(f'× Could not create platform config: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Platform configuration created"
    fi
}

# Setup IDE configurations
setup_ide_configs() {
    print_status "Setting up IDE configurations..."
    
    # VS Code settings
    if [ ! -d ".vscode" ]; then
        mkdir -p .vscode
        
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/htmlcov": true
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
EOF
        print_success "VS Code settings created"
    fi
}

# Run validation tests
run_validation() {
    print_status "Running validation tests..."
    
    source venv/bin/activate 2>/dev/null || true
    
    python3 << 'EOF' || print_warning "Validation failed"
import sys
sys.path.append('.')

print('\n🧪 Running platform validation...\n')

try:
    from src.platform_detector import PlatformDetector
    from config.platform_configs import get_platform_configs
    
    detector = PlatformDetector()
    info = detector.get_platform_info()
    config = get_platform_configs(detector)
    
    print('Platform Information:')
    print(f'  OS: {info["os"]} {info["os_version"]}')
    print(f'  Architecture: {info["architecture"]}')
    print(f'  Python: {info["python_version"]}')
    print(f'  Memory: {info["memory_gb"]}GB')
    print(f'  CPU Count: {info["cpu_count"]}')
    print(f'  Raspberry Pi: {info["is_raspberry_pi"]}')
    print(f'  Apple Silicon: {info["is_apple_silicon"]}')
    print()
    
    print('Configuration:')
    print(f'  Platform Type: {config["platform_type"]}')
    print(f'  Worker Processes: {config["worker_processes"]}')
    print(f'  Memory Limit: {config["memory_limit_mb"]}MB')
    print(f'  Face Detection Model: {config["face_detection_model"]}')
    print(f'  Installation Method: {config["installation_method"]}')
    print()
    
    print('✅ Platform validation successful!')
    
except Exception as e:
    print(f'❌ Platform validation failed: {e}')
    sys.exit(1)
EOF
}

# Display next steps
display_next_steps() {
    echo ""
    echo "=============================================="
    print_success "Development Environment Setup Complete! 🎉"
    echo "=============================================="
    echo ""
    print_status "Next steps:"
    echo ""
    echo "1. Activate virtual environment:"
    echo "   $ source venv/bin/activate"
    echo ""
    echo "2. Configure credentials (optional):"
    echo "   $ nano config/credentials_telegram.py"
    echo ""
    echo "3. Add known faces (optional):"
    echo "   $ cp <face_images> data/known_faces/"
    echo ""
    echo "4. Run tests:"
    echo "   $ python -m pytest tests/ -v"
    echo ""
    echo "5. Start the application:"
    case "$PLATFORM" in
        macos)
            echo "   $ ./scripts/start-web.sh"
            echo "   $ open http://localhost:5000"
            ;;
        raspberry_pi)
            echo "   $ sudo systemctl start doorbell-security"
            echo "   $ sudo systemctl status doorbell-security"
            ;;
        linux)
            echo "   $ ./scripts/start.sh"
            echo "   $ xdg-open http://localhost:5000"
            ;;
    esac
    echo ""
    print_status "📚 Documentation: README.md"
    print_status "🐛 Issues: https://github.com/itsnothuy/Doorbell-System/issues"
    print_status "💬 Discussions: https://github.com/itsnothuy/Doorbell-System/discussions"
    echo ""
}

# Main setup function
main() {
    echo ""
    print_status "Starting universal development environment setup..."
    echo ""
    
    # Check if in correct directory
    if [ ! -f "pyproject.toml" ] && [ ! -f "app.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Detect platform
    detect_platform
    
    # Run platform-specific installer
    run_platform_installer
    
    # Additional setup steps
    echo ""
    print_status "Performing additional setup..."
    
    setup_precommit
    create_platform_config
    setup_ide_configs
    run_validation
    
    # Display next steps
    display_next_steps
}

# Run main function
main "$@"
