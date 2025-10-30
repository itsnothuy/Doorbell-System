# Testing Scripts

This directory contains comprehensive testing utilities for the Doorbell Security System.

## Available Scripts

### `run_tests.py`

Unified test runner for all test types.

**Usage:**

```bash
# Run all tests
python scripts/testing/run_tests.py --all

# Run specific test suites
python scripts/testing/run_tests.py --unit --coverage
python scripts/testing/run_tests.py --integration --verbose
python scripts/testing/run_tests.py --e2e
python scripts/testing/run_tests.py --performance --benchmark
python scripts/testing/run_tests.py --security

# Quick mode (skip slow tests)
python scripts/testing/run_tests.py --all --quick

# Load testing
python scripts/testing/run_tests.py --load --users 50 --runtime 60
```

**Features:**
- Runs unit, integration, E2E, performance, security, and load tests
- Generates coverage reports (HTML, XML, JSON)
- Provides detailed test summaries
- Supports verbose output for debugging

### `generate_coverage_report.py`

Enhanced coverage report generator with multiple output formats.

**Usage:**

```bash
# Generate coverage report and run tests
python scripts/testing/generate_coverage_report.py

# Check coverage threshold
python scripts/testing/generate_coverage_report.py --fail-under 80

# Generate markdown report
python scripts/testing/generate_coverage_report.py --markdown

# Analyze existing coverage without running tests
python scripts/testing/generate_coverage_report.py --no-run --markdown
```

**Features:**
- Multiple output formats (text, markdown, JSON)
- Coverage threshold checking
- Package-level coverage breakdown
- Coverage badges for documentation

## Test Types

### Unit Tests

Test individual components in isolation.

```bash
pytest tests/ -m unit -v
```

### Integration Tests

Test component interactions and workflows.

```bash
pytest tests/integration/ -v
```

### End-to-End Tests

Test complete user journeys and system workflows.

```bash
# With Playwright
pytest tests/e2e/ -v --headed  # Show browser
pytest tests/e2e/ -v           # Headless

# Requires: pip install playwright pytest-playwright
# Setup: playwright install chromium
```

### Performance Tests

Benchmark and performance regression tests.

```bash
pytest tests/performance/ -v --benchmark-only
```

### Security Tests

Security-focused tests and vulnerability scanning.

```bash
pytest tests/security/ -v -m security
```

### Load Tests

Load and stress testing with Locust.

```bash
# Interactive mode
locust -f tests/load/locustfile.py --host=http://localhost:5000

# Headless mode
locust -f tests/load/locustfile.py --host=http://localhost:5000 \
       --headless --users 100 --spawn-rate 10 --run-time 120s \
       --html load-test-report.html
```

## Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only E2E tests
pytest -m e2e

# Run only performance tests
pytest -m performance

# Run only security tests
pytest -m security

# Run property-based tests
pytest -m property

# Exclude slow tests
pytest -m "not slow"

# Exclude hardware tests (when not on Raspberry Pi)
pytest -m "not hardware"
```

## Coverage Reports

### Generate Comprehensive Coverage

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov=config \
       --cov-report=html \
       --cov-report=xml \
       --cov-report=json \
       --cov-report=term-missing

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Thresholds

The project aims for 80%+ coverage. Check with:

```bash
python scripts/testing/generate_coverage_report.py --fail-under 80
```

## Continuous Integration

Tests run automatically on:
- Pull requests to main/develop
- Pushes to main/develop
- Nightly scheduled runs
- Manual workflow dispatch

See `.github/workflows/comprehensive-tests.yml` for the full CI/CD configuration.

## Property-Based Testing

Use Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st
from tests.utils.property_based_tests import valid_image_array

@given(valid_image_array())
def test_image_processing(image):
    result = process_image(image)
    assert result.shape == image.shape
```

Run property-based tests:

```bash
pytest tests/utils/property_based_tests.py -v --hypothesis-show-statistics
```

## Best Practices

1. **Write tests first**: Follow TDD when adding new features
2. **Use markers**: Tag tests appropriately (unit, integration, e2e, etc.)
3. **Mock external dependencies**: Use fixtures and mocks for hardware, APIs, etc.
4. **Keep tests fast**: Unit tests should run in milliseconds
5. **Clean test data**: Use fixtures to create and cleanup test data
6. **Test edge cases**: Include boundary conditions and error scenarios
7. **Document tests**: Add docstrings explaining what each test verifies

## Troubleshooting

### Tests Fail Due to Missing Dependencies

```bash
# Install all test dependencies
pip install -e '.[dev,testing,e2e,performance,security]'
```

### Playwright Tests Fail

```bash
# Install Playwright browsers
playwright install chromium
```

### Load Tests Can't Connect

Make sure the application is running:

```bash
# Terminal 1: Start application
python app.py

# Terminal 2: Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:5000
```

### Coverage Reports Missing

Ensure pytest-cov is installed:

```bash
pip install pytest-cov
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Playwright Documentation](https://playwright.dev/python/)
- [Locust Documentation](https://docs.locust.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
