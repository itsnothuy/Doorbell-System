# Windows Installation Script for Doorbell Security System
# PowerShell script for Windows 10/11

$ErrorActionPreference = "Continue"

Write-Host "🪟 Installing Doorbell Security System on Windows..." -ForegroundColor Cyan
Write-Host ""

# Function to print colored output
function Write-Status {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if running as Administrator
function Test-Admin {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check system requirements
function Test-SystemRequirements {
    Write-Status "Checking system requirements..."
    
    # Check Windows version
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-Error "Windows 10 or later is required"
        exit 1
    }
    Write-Success "Windows version: $($osVersion.ToString())"
    
    # Check available memory
    $memory = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    Write-Status "Available memory: $([math]::Round($memory, 2))GB"
    
    if ($memory -lt 2) {
        Write-Warning "Low memory detected. Installation may be slow."
    }
    
    Write-Success "System requirements met"
}

# Check for Python
function Test-Python {
    Write-Status "Checking for Python..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+\.\d+\.\d+)") {
            $version = $matches[1]
            Write-Success "Python $version found"
            
            # Check if version is 3.10+
            $versionParts = $version.Split('.')
            $major = [int]$versionParts[0]
            $minor = [int]$versionParts[1]
            
            if ($major -eq 3 -and $minor -ge 10) {
                return $true
            } elseif ($major -gt 3) {
                return $true
            } else {
                Write-Warning "Python 3.10+ is recommended (found $version)"
                return $true
            }
        }
    } catch {
        Write-Warning "Python not found in PATH"
        return $false
    }
    
    return $false
}

# Install Python if needed
function Install-Python {
    Write-Status "Python installation required"
    Write-Status "Please install Python 3.11 from: https://www.python.org/downloads/"
    Write-Status "Make sure to check 'Add Python to PATH' during installation"
    Write-Status ""
    Write-Status "After installing Python, run this script again."
    
    # Offer to open download page
    $response = Read-Host "Open Python download page in browser? (Y/N)"
    if ($response -eq 'Y' -or $response -eq 'y') {
        Start-Process "https://www.python.org/downloads/"
    }
    
    exit 1
}

# Check for Visual C++ Build Tools
function Test-BuildTools {
    Write-Status "Checking for Visual C++ Build Tools..."
    
    # Check for common build tools locations
    $buildToolsPaths = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community"
    )
    
    foreach ($path in $buildToolsPaths) {
        if (Test-Path $path) {
            Write-Success "Visual C++ Build Tools found"
            return $true
        }
    }
    
    Write-Warning "Visual C++ Build Tools not found"
    Write-Status "Some packages may require Visual C++ Build Tools"
    Write-Status "Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    
    return $false
}

# Create virtual environment
function New-VirtualEnvironment {
    Write-Status "Creating Python virtual environment..."
    
    if (Test-Path "venv") {
        Write-Warning "Virtual environment already exists"
        $response = Read-Host "Remove and recreate? (Y/N)"
        if ($response -eq 'Y' -or $response -eq 'y') {
            Remove-Item -Recurse -Force "venv"
        } else {
            Write-Status "Using existing virtual environment"
            return
        }
    }
    
    python -m venv venv
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Virtual environment created"
    } else {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

# Activate virtual environment
function Enter-VirtualEnvironment {
    Write-Status "Activating virtual environment..."
    
    $activateScript = "venv\Scripts\Activate.ps1"
    
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Success "Virtual environment activated"
    } else {
        Write-Error "Virtual environment activation script not found"
        exit 1
    }
}

# Install Python dependencies
function Install-PythonDependencies {
    Write-Status "Installing Python dependencies..."
    
    # Upgrade pip
    Write-Status "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install core dependencies
    Write-Status "Installing face recognition dependencies..."
    Write-Warning "This may take 10-20 minutes. Please be patient..."
    
    # Install cmake
    pip install cmake
    
    # Install dlib (can be problematic on Windows)
    Write-Status "Installing dlib..."
    pip install dlib
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "dlib installation failed"
        Write-Status "Trying alternative installation method..."
        
        # Try installing pre-built wheel if available
        pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.2-cp311-cp311-win_amd64.whl
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Could not install dlib. You may need Visual C++ Build Tools"
            Write-Status "See: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        }
    }
    
    # Install face_recognition
    Write-Status "Installing face_recognition..."
    pip install face_recognition
    
    # Install OpenCV
    Write-Status "Installing OpenCV..."
    pip install opencv-python
    
    # Install project dependencies
    if (Test-Path "requirements-web.txt") {
        Write-Status "Installing web requirements..."
        pip install -r requirements-web.txt
    }
    
    if (Test-Path "requirements.txt") {
        Write-Status "Installing project requirements..."
        pip install -r requirements.txt
    }
    
    # Install from pyproject.toml if available
    if (Test-Path "pyproject.toml") {
        Write-Status "Installing from pyproject.toml..."
        pip install -e ".[dev,monitoring]"
    }
    
    Write-Success "Python dependencies installed"
}

# Create project directories
function New-ProjectDirectories {
    Write-Status "Creating project directories..."
    
    $directories = @(
        "data",
        "data\known_faces",
        "data\blacklist_faces",
        "data\captures",
        "data\logs",
        "config"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    # Create __init__.py files
    if (Test-Path "src") {
        New-Item -ItemType File -Path "src\__init__.py" -Force -ErrorAction SilentlyContinue | Out-Null
    }
    if (Test-Path "config") {
        New-Item -ItemType File -Path "config\__init__.py" -Force -ErrorAction SilentlyContinue | Out-Null
    }
    
    Write-Success "Project directories created"
}

# Create configuration
function New-Configuration {
    Write-Status "Creating Windows configuration..."
    
    # Copy credentials template
    if ((Test-Path "config\credentials_template.py") -and -not (Test-Path "config\credentials_telegram.py")) {
        Copy-Item "config\credentials_template.py" "config\credentials_telegram.py"
        Write-Warning "Telegram credentials template copied to config\credentials_telegram.py"
        Write-Warning "Please edit this file with your actual bot token and chat ID"
    }
    
    # Create .env file
    if (-not (Test-Path ".env")) {
        $envContent = @"
# Doorbell Security System Environment Variables (Windows)

# Development mode
DEVELOPMENT_MODE=true

# Platform detection
PLATFORM_TYPE=windows

# Web interface
PORT=5000

# Camera Configuration
CAMERA_RESOLUTION=1280,720
CAMERA_ROTATION=0

# Face Recognition
FACE_TOLERANCE=0.6
BLACKLIST_TOLERANCE=0.5
REQUIRE_FACE_FOR_CAPTURE=False

# System
LOG_LEVEL=INFO
"@
        Set-Content -Path ".env" -Value $envContent
        Write-Success "Environment configuration created"
    } else {
        Write-Status "Environment file already exists, skipping..."
    }
}

# Create management scripts
function New-ManagementScripts {
    Write-Status "Creating management scripts..."
    
    if (-not (Test-Path "scripts")) {
        New-Item -ItemType Directory -Path "scripts" -Force | Out-Null
    }
    
    # Create start script
    $startScript = @"
# Start Doorbell Security System
Write-Host "🚀 Starting Doorbell Security System..." -ForegroundColor Cyan
& venv\Scripts\Activate.ps1
`$env:DEVELOPMENT_MODE = "true"
python app.py
"@
    Set-Content -Path "scripts\start.ps1" -Value $startScript
    
    # Create test script
    $testScript = @"
# Test Doorbell Security System
Write-Host "🧪 Testing Doorbell Security System on Windows..." -ForegroundColor Cyan
& venv\Scripts\Activate.ps1

python -c @"
import sys
sys.path.append('.')

from src.platform_detector import PlatformDetector
from config.platform_configs import get_platform_configs

print('🪟 Testing Windows platform detection...')
print()

detector = PlatformDetector()
info = detector.get_platform_info()

print('Platform Information:')
print(f'  OS: {info[""os""]} {info[""os_version""]}')
print(f'  Architecture: {info[""architecture""]}')
print(f'  Python: {info[""python_version""]}')
print(f'  Memory: {info[""memory_gb""]}GB')
print(f'  CPU Count: {info[""cpu_count""]}')
print()

print('Platform Configuration:')
config = get_platform_configs(detector)
print(f'  Platform Type: {config.get(""platform_type"")}')
print(f'  Worker Processes: {config.get(""worker_processes"")}')
print(f'  Memory Limit: {config.get(""memory_limit_mb"")}MB')
print()

print('✅ Platform detection test completed!')
"@

# Test imports
python -c @"
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
print('🎉 Windows test completed!')
print('💡 To start the application: .\scripts\start.ps1')
"@
"@
    Set-Content -Path "scripts\test-windows.ps1" -Value $testScript
    
    Write-Success "Management scripts created"
}

# Validate installation
function Test-Installation {
    Write-Status "Validating installation..."
    
    # Test Python version
    $pythonVersion = python --version 2>&1
    Write-Status "Python version: $pythonVersion"
    
    # Test imports
    python -c "import face_recognition" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "face_recognition: OK"
    } else {
        Write-Warning "face_recognition: NOT INSTALLED"
    }
    
    python -c "import cv2" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "opencv: OK"
    } else {
        Write-Warning "opencv: NOT INSTALLED"
    }
    
    python -c "import numpy" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "numpy: OK"
    } else {
        Write-Warning "numpy: NOT INSTALLED"
    }
    
    Write-Success "Validation complete"
}

# Main installation function
function Start-Installation {
    Write-Host ""
    Write-Status "Starting Doorbell Security System setup for Windows..."
    Write-Host ""
    
    # Check if in correct directory
    if (-not ((Test-Path "pyproject.toml") -or (Test-Path "app.py"))) {
        Write-Error "Please run this script from the project root directory"
        exit 1
    }
    
    # Run installation steps
    Test-SystemRequirements
    
    if (-not (Test-Python)) {
        Install-Python
    }
    
    Test-BuildTools
    New-VirtualEnvironment
    Enter-VirtualEnvironment
    Install-PythonDependencies
    New-ProjectDirectories
    New-Configuration
    New-ManagementScripts
    Test-Installation
    
    Write-Host ""
    Write-Success "Windows setup completed successfully! 🎉"
    Write-Host ""
    Write-Status "Next steps:"
    Write-Host "1. Edit config\credentials_telegram.py with your Telegram credentials (optional)"
    Write-Host "2. Add known face images to data\known_faces\ directory"
    Write-Host "3. Test the system: .\scripts\test-windows.ps1"
    Write-Host "4. Start the application: .\scripts\start.ps1"
    Write-Host "5. Open http://localhost:5000 in your browser"
    Write-Host ""
    Write-Status "📝 Notes for Windows:"
    Write-Host "   - Use Windows Defender exceptions if needed"
    Write-Host "   - Some features require Visual C++ Build Tools"
    Write-Host "   - Hardware GPIO and Pi Camera not supported"
    Write-Host ""
}

# Run main installation
Start-Installation
