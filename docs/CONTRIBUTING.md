# Contributing to Doorbell Security System

Thank you for your interest in contributing to the Doorbell Security System! This document provides guidelines and information for contributors.

## 🎯 Project Overview

The Doorbell Security System is an AI-powered security solution that combines face recognition, hardware integration, and smart notifications. Our goal is to create a privacy-first, locally-processed security system that works reliably across different platforms.

## 🌟 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new functionality
- **💻 Code Contributions**: Implement features or fix bugs
- **📚 Documentation**: Improve or add documentation
- **🧪 Testing**: Add tests or improve test coverage
- **🔒 Security**: Report security vulnerabilities
- **🎨 UI/UX**: Improve the web interface
- **🔧 Infrastructure**: Improve CI/CD, Docker, deployment

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/itsnothuy/doorbell-system.git
   cd doorbell-system
   ```

2. **Set Up Development Environment**
   
   **Option A: Using DevContainer (Recommended)**
   - Open in VS Code
   - Click "Reopen in Container" when prompted
   - Wait for automatic setup to complete
   
   **Option B: Local Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements-web.txt
   pip install -e ".[dev]"
   
   # Setup pre-commit hooks
   pre-commit install
   
   # Create necessary directories
   mkdir -p data/{known_faces,blacklist_faces,captures,logs,cropped_faces/{known,unknown}}
   ```

3. **Verify Setup**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run linting
   pre-commit run --all-files
   
   # Start application
   python app.py
   ```

## 🔄 Development Workflow

### Branch Strategy

- **`master`**: Main branch, always stable
- **`develop`**: Integration branch for new features
- **Feature branches**: `feature/description` or `fix/description`
- **Release branches**: `release/version`

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run full test suite
   pytest tests/ -v --cov=src --cov=config
   
   # Run linting
   pre-commit run --all-files
   
   # Test on multiple platforms if possible
   python app.py  # Test basic functionality
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new face recognition feature"
   ```
   
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `test:` - Test additions/modifications
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `ci:` - CI/CD changes

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a PR through GitHub interface.

## 📋 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters
- **Imports**: Use isort for import sorting
- **Formatting**: Use Black for code formatting
- **Type Hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

Our pre-commit hooks automatically run:

- **Ruff**: Fast Python linter
- **Black**: Code formatter
- **isort**: Import sorter
- **MyPy**: Type checking
- **Bandit**: Security linting

### Example Code Style

```python
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.platform_detector import platform_detector


class FaceDetector:
    """Detects faces in images using OpenCV and face_recognition.
    
    This class provides methods for detecting faces in images and extracting
    face encodings for recognition purposes.
    
    Attributes:
        tolerance: Recognition tolerance threshold
        model: Face detection model to use
    """
    
    def __init__(self, tolerance: float = 0.6, model: str = "hog") -> None:
        """Initialize the face detector.
        
        Args:
            tolerance: Face recognition tolerance (0.0-1.0)
            model: Detection model ('hog' or 'cnn')
        """
        self.tolerance = tolerance
        self.model = model
        self._initialize_detector()
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face detection results with bounding boxes and encodings
            
        Raises:
            ValueError: If image is invalid or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        # Implementation here...
        return []
```

### Documentation Standards

- **README**: Keep updated with setup instructions
- **Docstrings**: All public functions and classes
- **Type Hints**: All function parameters and return values
- **Comments**: Explain complex logic and business rules
- **ADRs**: Document architectural decisions in `docs/adr/`

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── performance/    # Performance and benchmark tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Unit Tests**: Test individual functions/classes in isolation
   ```python
   def test_face_detection_with_valid_image():
       detector = FaceDetector()
       image = create_test_image_with_face()
       results = detector.detect_faces(image)
       assert len(results) == 1
       assert results[0]['confidence'] > 0.8
   ```

2. **Integration Tests**: Test component interactions
   ```python
   def test_doorbell_press_workflow(mock_camera, mock_telegram):
       system = DoorbellSecuritySystem()
       system.on_doorbell_pressed(channel=18)
       # Verify the full workflow
   ```

3. **Performance Tests**: Use pytest-benchmark
   ```python
   def test_face_recognition_performance(benchmark):
       detector = FaceDetector()
       image = load_test_image()
       result = benchmark(detector.detect_faces, image)
       assert result is not None
   ```

### Test Requirements

- **Coverage**: Aim for >90% code coverage
- **Mocking**: Mock external dependencies (camera, GPIO, Telegram)
- **Fixtures**: Use pytest fixtures for test data
- **Performance**: Include performance regression tests
- **Cross-platform**: Test platform-specific code paths

## 🚀 Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No secrets or sensitive data in code

### PR Checklist

Use our [PR template](.github/PULL_REQUEST_TEMPLATE.md) which includes:

- **Description**: Clear explanation of changes
- **Testing**: How changes were tested
- **Breaking Changes**: Any compatibility issues
- **Security**: Security implications
- **Documentation**: Documentation updates

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review
3. **Testing**: Verify tests cover new functionality
4. **Documentation**: Ensure docs are updated
5. **Security**: Security review for sensitive changes

## 🎯 Component-Specific Guidelines

### Face Recognition (`src/face_manager.py`)

- **Privacy**: All processing must remain local
- **Performance**: Optimize for Raspberry Pi
- **Accuracy**: Test with diverse face data
- **Security**: Validate blacklist functionality

### Hardware Interface (`src/gpio_handler.py`, `src/camera_handler.py`)

- **Cross-platform**: Must work with mocks on non-Pi systems
- **Error Handling**: Graceful degradation when hardware unavailable
- **Threading**: Thread-safe operations
- **Resource Management**: Proper cleanup

### Web Interface (`src/web_interface.py`)

- **Security**: Input validation and sanitization
- **Responsive**: Mobile-friendly design
- **API**: RESTful design principles
- **Testing**: Include UI/API tests

### Configuration (`config/`)

- **Validation**: Validate all configuration inputs
- **Documentation**: Document all options
- **Backwards Compatibility**: Handle config migrations
- **Security**: Never log sensitive values

## 🐛 Bug Reports

### Before Reporting

1. **Search Existing Issues**: Check if already reported
2. **Update to Latest**: Test with latest version
3. **Minimal Reproduction**: Create minimal test case

### Bug Report Template

```markdown
**Describe the Bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Raspberry Pi OS, macOS, Ubuntu]
- Python Version: [e.g. 3.11.0]
- Hardware: [e.g. Raspberry Pi 4, x86_64]
- Version: [e.g. 1.0.0]

**Logs**
```
Paste relevant logs here
```

**Additional Context**
Any other context about the problem.
```

## 🔒 Security Guidelines

### Reporting Security Issues

**Do not report security vulnerabilities through public GitHub issues.**

Instead, email: [security@example.com] with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Best Practices

- **No Secrets**: Never commit API keys, passwords, or tokens
- **Input Validation**: Validate all user inputs
- **Dependencies**: Keep dependencies updated
- **Permissions**: Use minimal required permissions
- **Logging**: Don't log sensitive information

## 📚 Documentation Guidelines

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **User Documentation**: Setup and usage guides
3. **Developer Documentation**: Architecture and API docs
4. **ADRs**: Architectural Decision Records

### Writing Guidelines

- **Clear and Concise**: Use simple, direct language
- **Examples**: Include code examples where helpful
- **Structure**: Use consistent formatting and structure
- **Updates**: Keep documentation current with code changes

## 🎖️ Recognition

Contributors will be recognized in:
- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

## 📞 Getting Help

### Community Support

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: [maintainer@example.com] for direct contact

### Development Help

- **Documentation**: Check `/docs` directory
- **Examples**: See `/examples` directory
- **Tests**: Look at existing tests for patterns
- **Code**: Study existing implementations

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

## 🚀 Quick Reference

### Common Commands

```bash
# Setup
git clone https://github.com/itsnothuy/doorbell-system.git
cd doorbell-system
pip install -e ".[dev]"
pre-commit install

# Development
pytest tests/ -v                    # Run tests
pre-commit run --all-files         # Run linting
python app.py                      # Start application
docker-compose up                  # Run with Docker

# Code Quality
black src/ config/ tests/          # Format code
ruff check src/ config/ tests/     # Lint code
mypy src/ config/                  # Type checking
bandit -r src/ config/ app.py      # Security scan
```

### File Structure

```
doorbell-system/
├── src/                    # Main application code
├── config/                 # Configuration files
├── tests/                  # Test files
├── docs/                   # Documentation
├── .github/               # GitHub workflows and templates
├── data/                  # Data directories
├── static/                # Web interface assets
├── templates/             # Web interface templates
└── scripts/               # Utility scripts
```

Thank you for contributing to the Doorbell Security System! 🚪🔒