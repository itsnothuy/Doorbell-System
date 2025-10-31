# Cross-Platform Setup Guide

Complete installation and setup guide for the Doorbell Security System across all supported platforms.

## Table of Contents

- [Quick Start](#quick-start)
- [Platform-Specific Guides](#platform-specific-guides)
  - [macOS Setup](#macos-setup)
  - [Ubuntu/Debian Setup](#ubuntudebian-setup)
  - [Raspberry Pi Setup](#raspberry-pi-setup)
  - [Windows Setup](#windows-setup)
  - [Docker Setup](#docker-setup)
- [Platform Comparison](#platform-comparison)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Universal Setup Script

The easiest way to set up the Doorbell Security System is to use the universal setup script that automatically detects your platform and runs the appropriate installer:

```bash
# Clone the repository
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System

# Run universal setup
chmod +x scripts/setup_dev_environment.sh
./scripts/setup_dev_environment.sh
```

The script will:
1. Detect your platform (macOS, Linux, Raspberry Pi, Windows)
2. Run the platform-specific installer
3. Set up pre-commit hooks
4. Create platform-specific configuration
5. Validate the installation
6. Provide next steps

---

## Platform-Specific Guides

## macOS Setup

### System Requirements

- **macOS Version**: 10.15 Catalina or later
- **Architecture**: Intel (x86_64) or Apple Silicon (M1/M2/M3)
- **Memory**: 4GB RAM recommended (2GB minimum)
- **Disk Space**: 2GB free space

### Installation

#### Automatic Installation (Recommended)

```bash
# Run macOS installer
chmod +x scripts/install/install_macos.sh
./scripts/install/install_macos.sh
```

#### Manual Installation

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.11 cmake pkg-config jpeg libpng git

# Apple Silicon specific
if [[ $(uname -m) == "arm64" ]]; then
    export CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64"
    export ARCHFLAGS="-arch arm64"
fi

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install cmake dlib face_recognition opencv-python
pip install -e ".[dev,monitoring]"
```

### Platform-Specific Configuration

The macOS installation automatically configures:

- **Development Mode**: Enabled by default
- **GPIO Backend**: Mock (no hardware GPIO on macOS)
- **Camera Backend**: OpenCV (uses built-in webcam)
- **Face Detection Model**: HOG (CPU-efficient)
- **Metal Acceleration**: Enabled on Apple Silicon

### Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Start web interface
./scripts/start-web.sh

# Or run directly
python app.py
```

Access the web interface at: http://localhost:5000

### Testing

```bash
# Run platform detection test
./scripts/test-macos.sh

# Run full test suite
python -m pytest tests/ -v
```

### Apple Silicon Notes

- Some packages may need Rosetta 2 for compatibility
- Metal GPU acceleration is automatically enabled
- Face detection uses HOG model for best compatibility
- dlib may take longer to compile (15-20 minutes)

---

## Ubuntu/Debian Setup

### System Requirements

- **OS**: Ubuntu 20.04+ or Debian 11+
- **Architecture**: x86_64 (AMD64) or ARM64
- **Memory**: 2GB RAM recommended
- **Disk Space**: 3GB free space

### Installation

#### Automatic Installation (Recommended)

```bash
# Run Ubuntu installer
chmod +x scripts/install/install_ubuntu.sh
./scripts/install/install_ubuntu.sh
```

#### Manual Installation

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential cmake pkg-config \
    python3 python3-pip python3-venv python3-dev \
    libjpeg-dev libpng-dev libopenblas-dev \
    libboost-all-dev git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install cmake dlib face_recognition opencv-python
pip install -e ".[dev,monitoring]"
```

### Platform-Specific Configuration

The Ubuntu installation configures:

- **Development Mode**: Disabled (production mode)
- **GPIO Backend**: Mock (unless on Raspberry Pi)
- **Camera Backend**: OpenCV with V4L2
- **Face Detection Model**: CNN (can use GPU if available)
- **Worker Processes**: 2 (scales with CPU count)

### Running the Application

```bash
# Start application
./scripts/start.sh

# Or with systemd (production)
sudo systemctl start doorbell-security
sudo systemctl status doorbell-security
```

### Testing

```bash
# Run platform tests
./scripts/test-ubuntu.sh

# Run full test suite
python -m pytest tests/ -v
```

---

## Raspberry Pi Setup

### System Requirements

- **Model**: Raspberry Pi 3B+ or newer (Pi 4 recommended)
- **OS**: Raspberry Pi OS (Bullseye or later)
- **Memory**: 1GB RAM minimum (2GB+ recommended)
- **Disk Space**: 4GB free space
- **Camera**: Optional (Pi Camera Module or USB webcam)

### Installation

#### Automatic Installation (Recommended)

```bash
# Run Raspberry Pi installer
chmod +x scripts/install/install_raspberry_pi.sh
./scripts/install/install_raspberry_pi.sh
```

**Note**: Installation on Raspberry Pi takes 30-60 minutes due to compiling dlib from source.

#### Manual Installation

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential cmake \
    python3 python3-pip python3-venv python3-dev \
    libopenblas-dev libatlas-base-dev \
    python3-numpy python3-opencv python3-pil \
    python3-rpi.gpio python3-picamera2

# Create virtual environment (memory-optimized)
python3 -m venv venv
source venv/bin/activate

# Install with memory optimization
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=1

pip install --no-cache-dir cmake
pip install --no-cache-dir dlib
pip install --no-cache-dir face_recognition
pip install --no-cache-dir opencv-python-headless
pip install --no-cache-dir -e ".[pi,monitoring]"
```

### Memory Optimization

For Raspberry Pi with limited memory (<1GB):

```bash
# Increase swap size
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048

# Restart swap
sudo dphys-swapfile swapoff
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Platform-Specific Configuration

The Raspberry Pi installation configures:

- **Low Memory Mode**: Enabled
- **GPIO Backend**: RPi.GPIO (real hardware)
- **Camera Backend**: PiCamera2
- **Face Detection Model**: HOG (memory efficient)
- **Worker Processes**: 1 (memory constrained)
- **Memory Limit**: 1024MB

### Running as a Service

```bash
# Start service
sudo systemctl start doorbell-security

# Enable on boot
sudo systemctl enable doorbell-security

# Check status
sudo systemctl status doorbell-security

# View logs
sudo journalctl -u doorbell-security -f
```

### Testing

```bash
# Run platform tests
./scripts/test-pi.sh

# Test hardware
python -c "
import RPi.GPIO as GPIO
print('GPIO: OK')

from picamera2 import Picamera2
camera = Picamera2()
print('Camera: OK')
"
```

### Hardware Setup

1. **GPIO Connections**:
   - Doorbell Button: GPIO 18
   - Red LED: GPIO 16
   - Yellow LED: GPIO 20
   - Green LED: GPIO 21

2. **Camera**:
   - Enable camera: `sudo raspi-config` → Interface Options → Camera
   - Connect Pi Camera Module to CSI port
   - Or use USB webcam (/dev/video0)

---

## Windows Setup

### System Requirements

- **OS**: Windows 10 or Windows 11
- **Architecture**: x86_64 (AMD64)
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Disk Space**: 3GB free space
- **Python**: 3.10+ (will be installed if needed)

### Installation

#### Automatic Installation (Recommended)

Open PowerShell as Administrator:

```powershell
# Set execution policy
Set-ExecutionPolicy Bypass -Scope Process -Force

# Run Windows installer
.\scripts\install\install_windows.ps1
```

#### Manual Installation

```powershell
# Install Python 3.11 from python.org
# Make sure to check "Add Python to PATH"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install cmake dlib face_recognition opencv-python
pip install -e ".[dev,monitoring]"
```

### Visual C++ Build Tools

Some packages require Visual C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart PowerShell after installation

### Platform-Specific Configuration

The Windows installation configures:

- **Development Mode**: Enabled
- **GPIO Backend**: Mock (no hardware GPIO on Windows)
- **Camera Backend**: OpenCV (uses built-in webcam)
- **Face Detection Model**: HOG
- **Path Separator**: Backslash (\)

### Running the Application

```powershell
# Start application
.\scripts\start.ps1

# Or manually
.\venv\Scripts\Activate.ps1
python app.py
```

### Testing

```powershell
# Run platform tests
.\scripts\test-windows.ps1

# Run test suite
python -m pytest tests/unit/ -v
```

### Windows-Specific Notes

- Use Windows Defender exceptions for project directory if needed
- Some hardware features not supported (GPIO, Pi Camera)
- Face recognition may be slower than on Linux/macOS
- Use conda as an alternative package manager if pip fails

---

## Docker Setup

### System Requirements

- **Docker**: 20.10+ with BuildKit support
- **Docker Compose**: 2.0+
- **Architecture**: AMD64 or ARM64

### Quick Start

```bash
# Build and run with docker-compose
docker-compose up -d

# Or use multi-platform compose
docker-compose -f docker-compose.multiplatform.yml up -d
```

### Multi-Architecture Build

```bash
# Enable Docker Buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.multiarch \
  -t doorbell-security:latest \
  --push .
```

### Platform-Specific Containers

#### Raspberry Pi (ARM64)

```bash
docker-compose -f docker-compose.multiplatform.yml up doorbell-app-pi
```

Configuration:
- Memory Limit: 1GB
- CPU Limit: 1 core
- Platform: linux/arm64
- GPIO device access

#### Server (AMD64)

```bash
docker-compose -f docker-compose.multiplatform.yml up doorbell-app-server
```

Configuration:
- Memory Limit: 2GB
- CPU Limit: 2 cores
- Platform: linux/amd64
- CNN face detection

### Docker Configuration

Environment variables in `docker-compose.yml`:

```yaml
environment:
  - PLATFORM_MODE=auto  # or: raspberry_pi, server, development
  - LOG_LEVEL=INFO
  - FACE_TOLERANCE=0.6
  - MEMORY_LIMIT_MB=2048
```

### Volumes

Persistent data storage:

```yaml
volumes:
  - ./data:/app/data                    # Face images, captures
  - ./config:/app/config:ro             # Configuration files
  - ./logs:/app/data/logs               # Application logs
```

### Testing

```bash
# Test Docker image
docker run --rm doorbell-security:latest python -c "print('OK')"

# Check health
docker ps
docker inspect doorbell-app | grep -i health
```

---

## Platform Comparison

| Feature | macOS | Ubuntu | Raspberry Pi | Windows | Docker |
|---------|-------|---------|--------------|---------|--------|
| **Installation Time** | 15-20 min | 20-30 min | 30-60 min | 20-30 min | 10-15 min |
| **Memory Required** | 2-4GB | 2GB | 1GB | 2GB | 1-2GB |
| **Face Detection** | HOG/CNN | CNN | HOG | HOG | HOG |
| **GPU Support** | Metal (M1/M2) | CUDA/ROCm | No | Limited | No |
| **GPIO Support** | Mock | Mock | Real | Mock | Mock |
| **Camera Support** | Webcam | Webcam/USB | Pi Camera | Webcam | USB |
| **Auto-start** | No | Systemd | Systemd | No | Docker restart |
| **Production Ready** | Dev Only | Yes | Yes | No | Yes |

---

## Troubleshooting

### Common Issues

#### dlib Installation Fails

**Symptoms**: `error: command 'gcc' failed` or compilation errors

**Solution**:
```bash
# macOS
brew install cmake
export CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=$(uname -m)"

# Ubuntu
sudo apt install build-essential cmake libboost-all-dev

# Raspberry Pi
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=1
pip install --no-cache-dir --verbose dlib
```

#### face_recognition Import Error

**Symptoms**: `ModuleNotFoundError: No module named 'face_recognition'`

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows

# Reinstall
pip install --force-reinstall face_recognition
```

#### Memory Issues on Raspberry Pi

**Symptoms**: Installation crashes or "Killed" messages

**Solution**:
```bash
# Increase swap
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048

# Restart swap
sudo dphys-swapfile swapoff && sudo dphys-swapfile setup && sudo dphys-swapfile swapon

# Use memory-optimized installation
export PIP_NO_CACHE_DIR=1
export MAX_JOBS=1
```

#### Camera Not Detected

**Symptoms**: `Unable to open camera` or camera errors

**Solution**:
```bash
# macOS - Grant camera permissions in System Preferences

# Ubuntu - Check V4L2 devices
ls -la /dev/video*

# Raspberry Pi - Enable camera interface
sudo raspi-config
# Interface Options → Camera → Enable

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

#### Windows Visual C++ Build Tools Missing

**Symptoms**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart terminal and try again
4. Alternative: Use pre-built wheels or conda

### Platform-Specific Troubleshooting

For more detailed troubleshooting, see:
- [macOS Troubleshooting](TROUBLESHOOTING_MACOS.md)
- [Ubuntu Troubleshooting](TROUBLESHOOTING_UBUNTU.md)
- [Raspberry Pi Troubleshooting](TROUBLESHOOTING_PI.md)
- [Windows Troubleshooting](TROUBLESHOOTING_WINDOWS.md)

---

## Support

- **GitHub Issues**: https://github.com/itsnothuy/Doorbell-System/issues
- **Discussions**: https://github.com/itsnothuy/Doorbell-System/discussions
- **Documentation**: https://github.com/itsnothuy/Doorbell-System/wiki

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Last Updated**: 2024-10-31
**Version**: 1.0.0
