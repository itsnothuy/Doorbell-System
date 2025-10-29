# Testing Framework Documentation

## Overview

Comprehensive testing framework for the Doorbell Security System with 95%+ code coverage target, organized following the testing pyramid architecture.

## Test Structure

```
tests/
├── conftest.py                    # Enhanced pytest configuration with fixtures
├── fixtures/                      # Specialized test fixtures
│   ├── hardware_fixtures.py       # Hardware mocking (camera, GPIO)
│   ├── pipeline_fixtures.py       # Pipeline component fixtures
│   ├── data_fixtures.py          # Test data generation
│   └── performance_fixtures.py    # Performance monitoring fixtures
├── utils/                         # Testing utilities
│   ├── test_helpers.py           # Common helper functions
│   ├── mock_hardware.py          # Comprehensive hardware mocks
│   └── test_data_generator.py    # Test data generation utilities
├── unit/                          # Unit tests (60% of tests)
├── integration/                   # Integration tests (30% of tests)
├── e2e/                          # End-to-end tests (10% of tests)
├── performance/                   # Performance benchmarks
├── security/                      # Security validation tests
└── load/                         # Load and stress tests
```

## Testing Pyramid

```
           /\
          /E2E\        10% - Complete system scenarios
         /____\
        /      \
       /Integr.\      30% - Component interactions
      /________\
     /          \
    /Unit Tests  \    60% - Individual components
   /______________\
```

## Quick Start

### Install Dependencies

```bash
# Install all testing dependencies
pip install -e '.[dev,testing]'
```

### Validate Test Environment

```bash
# Check that all test requirements are met
python scripts/testing/validate_test_environment.py
```

### Run Tests

```bash
# Run all tests with coverage
python scripts/testing/run_full_test_suite.py --coverage

# Run specific test categories
python scripts/testing/run_full_test_suite.py --unit
python scripts/testing/run_full_test_suite.py --integration
python scripts/testing/run_full_test_suite.py --e2e

# Run with parallel execution
python scripts/testing/run_full_test_suite.py --parallel 4
```

## Test Categories

### Unit Tests (60%)

Fast, isolated tests for individual components.

```bash
# Run unit tests only
pytest tests/unit/ -v

# Run specific unit test file
pytest tests/unit/test_pipeline/test_frame_capture.py -v
```

**What to test:**
- Individual functions and methods
- Class initialization and state
- Error handling and edge cases
- Input validation
- Return value correctness

### Integration Tests (30%)

Tests for component interactions and data flow.

```bash
# Run integration tests
pytest tests/integration/ -v
```

**What to test:**
- Pipeline component interactions
- Message bus communication
- Database operations
- Hardware integration
- API endpoints

### End-to-End Tests (10%)

Complete system scenarios from trigger to notification.

```bash
# Run e2e tests
pytest tests/e2e/ -v -m e2e
```

**What to test:**
- Complete doorbell trigger flows
- Face recognition end-to-end
- Error recovery scenarios
- User scenarios

### Performance Tests

Benchmarks and performance validation.

```bash
# Run performance tests
python scripts/testing/run_performance_tests.py

# Run specific benchmark
python scripts/testing/run_performance_tests.py --benchmark throughput
```

**What to test:**
- Throughput (frames per second)
- Latency (end-to-end timing)
- Resource usage (CPU, memory)
- Scalability under load

### Security Tests

Security validation and vulnerability testing.

```bash
# Run security tests
pytest tests/security/ -v -m security
```

**What to test:**
- Input validation and sanitization
- SQL injection prevention
- Path traversal prevention
- Command injection prevention
- Authentication and authorization
- Data protection

### Load Tests

Stress testing and system limits.

```bash
# Run load tests
pytest tests/load/ -v -m slow
```

**What to test:**
- Concurrent user handling
- Sustained high load
- System limits
- Resource cleanup
- Memory leak detection

## Test Fixtures

### Hardware Fixtures

Mock hardware components for testing without physical hardware.

```python
def test_camera_capture(mock_camera):
    """Example using camera fixture."""
    frame = mock_camera.capture_array()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
```

### Pipeline Fixtures

Mock pipeline components for testing workflows.

```python
def test_pipeline_flow(mock_pipeline_workers):
    """Example using pipeline workers."""
    workers = mock_pipeline_workers
    assert workers['frame_capture'] is not None
```

### Performance Fixtures

Monitor and validate performance requirements.

```python
def test_latency(performance_monitor):
    """Example using performance monitor."""
    performance_monitor.start_timer('operation')
    # ... perform operation ...
    performance_monitor.end_timer('operation')
    
    metrics = performance_monitor.get_metrics()
    assert metrics['operation_duration'] < 1.0
```

## Test Markers

Use pytest markers to categorize and filter tests:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.e2e          # End-to-end test
@pytest.mark.performance  # Performance test
@pytest.mark.security     # Security test
@pytest.mark.slow         # Slow-running test
@pytest.mark.hardware     # Requires hardware
@pytest.mark.gpu          # Requires GPU
```

Run tests by marker:
```bash
pytest -m unit            # Run only unit tests
pytest -m "not slow"      # Skip slow tests
pytest -m "integration and not hardware"  # Integration without hardware
```

## Coverage Requirements

Target: **95%+ code coverage**

### Generate Coverage Report

```bash
# Generate comprehensive coverage report
python scripts/testing/generate_coverage_report.py --badge

# View HTML report
open htmlcov/index.html
```

### Coverage Configuration

Coverage settings in `pyproject.toml`:
- Minimum coverage: 80%
- Branch coverage: enabled
- HTML, XML, and JSON reports
- Excludes test files and third-party code

## Performance Requirements

Standard performance requirements for testing:

| Metric | Threshold |
|--------|-----------|
| Frame capture | < 100ms |
| Motion detection | < 50ms |
| Face detection | < 500ms |
| Face recognition | < 300ms |
| Event processing | < 100ms |
| End-to-end latency | < 2s |
| Throughput | > 10 FPS |
| Memory usage | < 500MB |
| CPU usage | < 80% |

## Writing Tests

### Test Structure

```python
#!/usr/bin/env python3
"""
Test module docstring describing what is tested.
"""

import pytest
from typing import Any


class TestComponentName:
    """Test suite for ComponentName."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        expected = "result"
        
        # Act
        actual = function_under_test()
        
        # Assert
        assert actual == expected
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_that_should_raise()
```

### Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion focus**: Test one thing at a time
3. **Descriptive names**: Test names should describe what they test
4. **Use fixtures**: Leverage fixtures for setup
5. **Mock external dependencies**: Use mocks for external services
6. **Test edge cases**: Include boundary conditions
7. **Performance awareness**: Monitor test execution time

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main/develop branches
- Nightly builds

### CI Configuration

See `.github/workflows/tests.yml` for full CI configuration.

Quality gates:
- All tests must pass
- Coverage must be ≥ 80%
- No security vulnerabilities
- Code style compliance

## Troubleshooting

### Tests Not Found

```bash
# Verify test discovery
pytest --collect-only
```

### Import Errors

```bash
# Ensure package is installed
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Slow Tests

```bash
# Run with parallel execution
pytest -n 4

# Skip slow tests
pytest -m "not slow"
```

### Memory Issues

```bash
# Monitor memory usage
pytest tests/ --memray
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)

## Support

For testing questions or issues:
1. Check this documentation
2. Review existing tests for examples
3. Open an issue on GitHub
4. Contact the development team
