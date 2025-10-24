#!/bin/bash
# DevContainer setup script for Doorbell Security System
# This script runs after the container is created to set up the development environment

set -e

echo "🚀 Setting up Doorbell Security System development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install additional system dependencies for face recognition and OpenCV
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1-dev \
    libfreetype6-dev \
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
    libatlas-base-dev \
    gfortran \
    python3-dev

# Install Python package management tools
echo "🐍 Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel

# Install the project in development mode
echo "📋 Installing project dependencies..."
if [ -f "requirements-web.txt" ]; then
    pip install -r requirements-web.txt
fi

if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev]"
fi

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if command -v pre-commit >/dev/null 2>&1; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo "✅ Pre-commit hooks installed"
else
    pip install pre-commit
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo "✅ Pre-commit installed and configured"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{known_faces,blacklist_faces,captures,logs,cropped_faces/{known,unknown}}
mkdir -p tests/{unit,integration,performance}
mkdir -p docs/{images,diagrams}
mkdir -p scripts
mkdir -p static/{css,js,images}

# Set proper permissions
sudo chown -R vscode:vscode /workspaces/doorbell-system
chmod -R 755 /workspaces/doorbell-system

# Create test data for development
echo "🧪 Creating test data..."
python -c "
import os
from PIL import Image
import numpy as np

# Create sample test images
def create_test_image(name, size=(200, 200)):
    img_data = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    img = Image.fromarray(img_data)
    return img

# Create known faces
known_faces = ['john_doe', 'jane_smith', 'bob_wilson']
for name in known_faces:
    img = create_test_image(name)
    img.save(f'data/known_faces/{name}.jpg')

# Create blacklist face
blacklist_img = create_test_image('suspicious_person')
blacklist_img.save('data/blacklist_faces/suspicious_person.jpg')

print('✅ Test data created')
"

# Setup Git configuration
echo "🔧 Configuring Git..."
if [ ! -f ~/.gitconfig ]; then
    git config --global user.name "Developer"
    git config --global user.email "developer@example.com"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
fi

# Install additional development tools
echo "🛠️ Installing development tools..."
pip install --upgrade \
    jupyter \
    ipython \
    notebook \
    jupyterlab \
    httpie \
    rich \
    click

# Install security scanning tools
echo "🔒 Installing security tools..."
pip install --upgrade \
    bandit \
    safety \
    semgrep

# Setup shell aliases and functions
echo "⚡ Setting up shell aliases..."
cat >> ~/.zshrc << 'EOF'

# Doorbell Security System aliases
alias ds-run="python app.py"
alias ds-test="pytest tests/ -v"
alias ds-lint="pre-commit run --all-files"
alias ds-format="black src/ config/ tests/ && isort src/ config/ tests/"
alias ds-type="mypy src/ config/"
alias ds-security="bandit -r src/ config/ app.py"
alias ds-deps="safety check"
alias ds-build="docker build -t doorbell-security ."
alias ds-compose="docker-compose up -d"
alias ds-logs="docker-compose logs -f"
alias ds-shell="docker-compose exec doorbell-security bash"

# Development helpers
alias ll="ls -la"
alias la="ls -la"
alias ..="cd .."
alias ...="cd ../.."
alias grep="grep --color=auto"
alias pip-upgrade="pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U"

# Git aliases
alias gs="git status"
alias ga="git add"
alias gc="git commit"
alias gp="git push"
alias gl="git log --oneline -10"
alias gd="git diff"
alias gb="git branch"
alias gco="git checkout"

# Python development
alias py="python"
alias ipy="ipython"
alias pym="python -m"
alias serve="python -m http.server"

# Quick test functions
function test-face() {
    echo "Testing face recognition system..."
    python -c "
    from src.face_manager import FaceManager
    fm = FaceManager()
    print('✅ Face manager imported successfully')
    "
}

function test-camera() {
    echo "Testing camera system..."
    python -c "
    from src.camera_handler import CameraHandler
    ch = CameraHandler()
    print('✅ Camera handler imported successfully')
    "
}

function test-system() {
    echo "Testing full system..."
    python -c "
    from src.doorbell_security import DoorbellSecuritySystem
    system = DoorbellSecuritySystem()
    print('✅ System initialized successfully')
    "
}

function ds-help() {
    echo "🚪 Doorbell Security System - Development Commands"
    echo ""
    echo "Running:"
    echo "  ds-run      - Start the application"
    echo "  ds-compose  - Start with Docker Compose"
    echo ""
    echo "Testing:"
    echo "  ds-test     - Run all tests"
    echo "  test-face   - Test face recognition"
    echo "  test-camera - Test camera system"
    echo "  test-system - Test full system"
    echo ""
    echo "Code Quality:"
    echo "  ds-lint     - Run all linters"
    echo "  ds-format   - Format code"
    echo "  ds-type     - Type checking"
    echo "  ds-security - Security scan"
    echo ""
    echo "Development:"
    echo "  ds-build    - Build Docker image"
    echo "  ds-logs     - View logs"
    echo "  ds-shell    - Enter container shell"
}

EOF

# Create useful development scripts
echo "📝 Creating development scripts..."
mkdir -p scripts

cat > scripts/dev-setup.sh << 'EOF'
#!/bin/bash
# Quick development setup script
set -e

echo "🔧 Setting up development environment..."

# Install dependencies
pip install -r requirements-web.txt
pip install -e ".[dev]"

# Setup pre-commit
pre-commit install

# Create directories
mkdir -p data/{known_faces,blacklist_faces,captures,logs,cropped_faces/{known,unknown}}

echo "✅ Development setup complete!"
EOF

cat > scripts/lint.sh << 'EOF'
#!/bin/bash
# Comprehensive linting script
set -e

echo "🧹 Running code quality checks..."

echo "📝 Running ruff..."
ruff check src/ config/ tests/ app.py

echo "🎨 Running black..."
black --check src/ config/ tests/ app.py

echo "📦 Running isort..."
isort --check-only src/ config/ tests/ app.py

echo "🔍 Running mypy..."
mypy src/ config/ app.py

echo "🔒 Running bandit..."
bandit -r src/ config/ app.py

echo "✅ All linting checks passed!"
EOF

cat > scripts/test.sh << 'EOF'
#!/bin/bash
# Comprehensive testing script
set -e

echo "🧪 Running test suite..."

# Set environment variables
export DEVELOPMENT_MODE=true
export PYTHONPATH=$(pwd)

echo "📋 Running unit tests..."
pytest tests/unit/ -v --cov=src --cov=config

echo "🔗 Running integration tests..."
pytest tests/integration/ -v

echo "⚡ Running performance tests..."
pytest tests/performance/ -v --benchmark-only

echo "✅ All tests passed!"
EOF

cat > scripts/run-local.sh << 'EOF'
#!/bin/bash
# Run the application locally
set -e

export DEVELOPMENT_MODE=true
export PYTHONPATH=$(pwd)

echo "🚪 Starting Doorbell Security System..."
python app.py
EOF

# Make scripts executable
chmod +x scripts/*.sh

# Install VS Code extensions via CLI (if code command is available)
if command -v code >/dev/null 2>&1; then
    echo "📦 Installing VS Code extensions..."
    code --install-extension ms-python.python
    code --install-extension charliermarsh.ruff
    code --install-extension ms-python.black-formatter
    code --install-extension github.copilot
    code --install-extension github.copilot-chat
fi

# Final setup verification
echo "🔍 Verifying setup..."
python --version
pip --version
git --version

# Test imports
python -c "
try:
    import cv2
    import PIL
    import numpy
    import face_recognition
    import flask
    import requests
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Doorbell Security System development environment setup complete!"
echo ""
echo "Quick start:"
echo "  1. Run 'ds-help' to see available commands"
echo "  2. Run 'ds-test' to run tests"
echo "  3. Run 'ds-run' to start the application"
echo ""
echo "Happy coding! 🚪🔒"