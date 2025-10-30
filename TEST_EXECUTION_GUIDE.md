# Test Suite Execution Guide

## Quick Start

### Run All Unit Tests
```bash
# Run all new unit tests
python3 -m pytest tests/unit/ -v

# Run with coverage
python3 -m pytest tests/unit/ --cov=src --cov=config --cov-report=term-missing
```

### Run Specific Test Files
```bash
# Test logging configuration
python3 -m pytest tests/unit/test_logging_config.py -v

# Test storage configuration
python3 -m pytest tests/unit/test_storage_config.py -v

# Test platform detector
python3 -m pytest tests/unit/test_platform_detector.py -v
```

### Run All Working Tests (Excluding Broken Imports)
```bash
python3 -m pytest tests/ \
  --ignore=tests/test_system.py \
  --ignore=tests/test_orchestrator_framework.py \
  --ignore=tests/test_sensor_integration.py \
  --ignore=tests/e2e \
  --ignore=tests/integration \
  --ignore=tests/streaming \
  -v
```

### Generate Coverage Report
```bash
# Generate HTML coverage report
python3 -m pytest tests/unit/ --cov=src --cov=config --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Dependencies**: Mocked
- **Speed**: Fast (< 5 seconds for all unit tests)
- **Files**: 11 test files, 82 tests
- **Status**: ✅ All passing

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions
- **Dependencies**: Real components, mocked external services
- **Speed**: Medium (30-60 seconds)
- **Files**: 14 test files
- **Status**: ⚠️ Import errors (requires face_recognition)

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Dependencies**: Full system running
- **Speed**: Slow (2-5 minutes)
- **Files**: Multiple test files
- **Status**: ⚠️ Import errors (requires Playwright)

### Performance Tests (`tests/performance/`)
- **Purpose**: Benchmark performance metrics
- **Dependencies**: System resources
- **Speed**: Variable
- **Status**: ✅ Mostly working

### Security Tests (`tests/security/`)
- **Purpose**: Validate security requirements
- **Dependencies**: Security scanners
- **Speed**: Medium
- **Status**: ✅ Mostly working

## Test Organization

```
tests/
├── unit/                          # New unit tests (82 tests)
│   ├── test_logging_config.py     # 14 tests ✅
│   ├── test_storage_config.py     # 27 tests ✅
│   ├── test_credentials_template.py # 7 tests ✅
│   ├── test_platform_detector.py  # 20 tests ✅
│   └── test_*_config.py          # 14 tests ✅
├── conftest.py                    # Pytest configuration and fixtures
├── baselines/                     # Performance baselines
├── fixtures/                      # Test data and fixtures
├── framework/                     # Test framework utilities
├── integration/                   # Integration tests (⚠️ imports)
├── e2e/                          # End-to-end tests (⚠️ imports)
├── performance/                   # Performance tests
├── security/                      # Security tests
├── streaming/                     # Streaming tests (⚠️ imports)
├── load/                         # Load tests
└── [50+ test files]              # Existing tests
```

## CI/CD Integration

### GitHub Actions Workflows

The project uses comprehensive CI/CD workflows:

```yaml
# .github/workflows/comprehensive-tests.yml
jobs:
  - code-quality          # Linting and formatting
  - unit-tests           # Python 3.10, 3.11, 3.12
  - integration-tests    # Component interaction tests
  - e2e-tests           # End-to-end workflow tests
  - performance-tests    # Benchmark and performance
  - load-tests          # Load testing with Locust
  - security-tests      # Security scanning
  - coverage-report     # Aggregate coverage report
  - quality-gates       # Final quality checks
```

### Running CI Tests Locally

```bash
# Code quality checks
black --check src config tests
ruff check src config tests
isort --check-only src config tests

# Run tests like CI
python3 -m pytest tests/ -v \
  --cov=src --cov=config \
  --cov-report=xml \
  --cov-report=html \
  --junit-xml=pytest-results.xml
```

## Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only unit tests
python3 -m pytest -m unit

# Run only integration tests
python3 -m pytest -m integration

# Run only performance tests
python3 -m pytest -m performance

# Run only security tests
python3 -m pytest -m security

# Skip slow tests
python3 -m pytest -m "not slow"

# Run hardware tests (requires hardware)
python3 -m pytest -m hardware
```

## Coverage Goals

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| config/ | ~15% | 95% | High |
| src/pipeline/ | ~5% | 95% | Critical |
| src/communication/ | ~40% | 95% | High |
| src/storage/ | ~10% | 95% | High |
| src/hardware/ | ~10% | 90% | Medium |
| src/detectors/ | ~45% | 95% | High |
| src/enrichment/ | ~30% | 90% | Medium |

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'face_recognition'`

**Solution**: Install face_recognition (requires cmake and dlib)
```bash
# macOS
brew install cmake
pip install face_recognition

# Ubuntu/Debian
sudo apt-get install cmake libboost-all-dev
pip install face_recognition

# Or use system package
sudo apt-get install python3-face-recognition
```

### Slow Tests

**Problem**: Tests take too long to run

**Solution**: Run specific test categories
```bash
# Run only fast unit tests
python3 -m pytest tests/unit/ -v

# Skip slow tests
python3 -m pytest -m "not slow"

# Run in parallel (requires pytest-xdist)
python3 -m pytest -n auto
```

### Coverage Not Updating

**Problem**: Coverage report not showing changes

**Solution**: Clear coverage cache
```bash
# Remove old coverage data
rm -rf .coverage htmlcov/ coverage.xml

# Run tests again
python3 -m pytest --cov=src --cov=config --cov-report=html
```

## Best Practices

### Writing New Tests

1. **Follow naming conventions**: `test_<module>_<function>.py`
2. **Use descriptive test names**: Explain what is being tested
3. **Include docstrings**: Describe test purpose
4. **Test edge cases**: Not just happy path
5. **Mock external dependencies**: Keep tests isolated
6. **Use fixtures**: Reuse test setup code
7. **Assert clearly**: Verify expected behavior

### Example Test Structure

```python
#!/usr/bin/env python3
"""
Unit tests for example_module.

Tests example functionality and edge cases.
"""

import pytest
from unittest.mock import Mock, patch

from src.example_module import ExampleClass


class TestExampleClass:
    """Test ExampleClass functionality."""
    
    @pytest.fixture
    def example_instance(self):
        """Create test instance."""
        return ExampleClass(config={'key': 'value'})
    
    def test_normal_operation(self, example_instance):
        """Test normal operation."""
        result = example_instance.method()
        assert result is not None
        assert result == expected_value
    
    def test_edge_case(self, example_instance):
        """Test edge case handling."""
        result = example_instance.method(edge_case_input)
        assert result handles edge case correctly
    
    def test_error_handling(self, example_instance):
        """Test error handling."""
        with pytest.raises(ExpectedException):
            example_instance.method(invalid_input)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

## Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Project CI/CD**: `.github/workflows/comprehensive-tests.yml`
- **Test Summary**: `COMPREHENSIVE_TEST_SUITE_SUMMARY.md`

## Support

For questions or issues:
1. Check this guide
2. Review existing test examples in `tests/unit/`
3. Consult `COMPREHENSIVE_TEST_SUITE_SUMMARY.md`
4. Review GitHub Actions workflow results
