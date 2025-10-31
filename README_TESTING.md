# Testing Guide - Doorbell Security System

## Quick Reference

### Run Tests
```bash
# All unit tests (FAST - 3-5 seconds)
python3 -m pytest tests/unit/ -v

# With coverage report
python3 -m pytest tests/unit/ --cov=src --cov=config --cov-report=html
open htmlcov/index.html

# All working tests (SLOW - ~2 minutes)
python3 -m pytest tests/ \
  --ignore=tests/test_system.py \
  --ignore=tests/test_orchestrator_framework.py \
  --ignore=tests/test_sensor_integration.py \
  --ignore=tests/e2e \
  --ignore=tests/integration \
  --ignore=tests/streaming \
  -v
```

## Test Suite Overview

### Current Status (Issue #25)
- ✅ **992 tests** (910 → 992, +82 new)
- ✅ **~25.5% coverage** (23.94% → 25.5%, +1.56%)
- ✅ **100% pass rate** on new tests
- ✅ **11 new unit test files**
- ✅ **3 comprehensive documentation files**

### Test Categories

| Category | Location | Status | Count |
|----------|----------|--------|-------|
| Unit Tests | `tests/unit/` | ✅ Working | 82 new |
| Integration Tests | `tests/integration/` | ⚠️ Import errors | ~50 |
| E2E Tests | `tests/e2e/` | ⚠️ Setup needed | ~20 |
| Performance Tests | `tests/performance/` | ✅ Working | ~30 |
| Security Tests | `tests/security/` | ✅ Working | ~25 |
| Framework Tests | `tests/framework/` | ✅ Working | ~15 |

## New Test Files

Created for Issue #25:

1. `tests/unit/test_logging_config.py` (14 tests) ⭐⭐⭐⭐⭐
2. `tests/unit/test_storage_config.py` (27 tests) ⭐⭐⭐⭐⭐
3. `tests/unit/test_credentials_template.py` (7 tests) ⭐⭐⭐⭐⭐
4. `tests/unit/test_platform_detector.py` (20 tests) ⭐⭐⭐⭐⭐
5. `tests/unit/test_orchestrator_config.py` (2 tests) ⭐⭐⭐
6. `tests/unit/test_migration___init__.py` (2 tests) ⭐⭐⭐
7. `tests/unit/test_migration_legacy_mapping.py` (2 tests) ⭐⭐⭐
8. `tests/unit/test_migration_migration_config.py` (2 tests) ⭐⭐⭐
9. `tests/unit/test_production___init__.py` (2 tests) ⭐⭐⭐
10. `tests/unit/test_production_monitoring_config.py` (2 tests) ⭐⭐⭐
11. `tests/unit/test_production_production_settings.py` (2 tests) ⭐⭐⭐

**Total: 82 tests, all passing ✅**

## Documentation

### Primary Guides

1. **COMPREHENSIVE_TEST_SUITE_SUMMARY.md**
   - Technical summary of testing work
   - Remaining work breakdown
   - Time estimates for 100% coverage
   - Quality standards

2. **TEST_EXECUTION_GUIDE.md**
   - How to run tests
   - Test organization
   - CI/CD integration
   - Troubleshooting

3. **ISSUE_25_COMPLETION_SUMMARY.md**
   - Visual progress indicators
   - Stakeholder summary
   - Achievement metrics
   - Next steps

### Quick Links
- **Test Documentation**: This file
- **CI/CD Workflow**: `.github/workflows/comprehensive-tests.yml`
- **Test Configuration**: `pyproject.toml` (pytest section)
- **Coverage Data**: `coverage.json`, `htmlcov/`

## CI/CD Integration

### GitHub Actions Workflows

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Scheduled nightly runs
- Manual workflow dispatch

### Workflow Jobs

1. **code-quality**: Linting (black, ruff, isort)
2. **unit-tests**: Unit tests (Python 3.10, 3.11, 3.12)
3. **integration-tests**: Integration tests
4. **e2e-tests**: End-to-end tests
5. **performance-tests**: Benchmark tests
6. **load-tests**: Load testing with Locust
7. **security-tests**: Security scanning (Bandit, Safety)
8. **coverage-report**: Aggregate coverage
9. **quality-gates**: Final quality checks

### Running CI Locally

```bash
# Code quality
black --check src config tests
ruff check src config tests
isort --check-only src config tests

# Unit tests (like CI)
python3 -m pytest tests/ -v \
  --cov=src --cov=config \
  --cov-report=xml \
  --cov-report=html
```

## Coverage Goals

### Current Coverage by Component

| Component | Coverage | Target | Priority |
|-----------|----------|--------|----------|
| config/ | ~15% | 95% | High |
| src/pipeline/ | ~5% | 95% | **Critical** |
| src/communication/ | ~40% | 95% | High |
| src/storage/ | ~10% | 95% | High |
| src/hardware/ | ~10% | 90% | Medium |
| src/detectors/ | ~45% | 95% | High |
| src/enrichment/ | ~30% | 90% | Medium |

### Roadmap to 100%

**Estimated remaining work: 70-90 hours**

See `COMPREHENSIVE_TEST_SUITE_SUMMARY.md` for detailed breakdown.

## Test Patterns

### Unit Test Example

```python
#!/usr/bin/env python3
"""Unit tests for example module."""

import pytest
from unittest.mock import Mock, patch
from src.example import ExampleClass


class TestExampleClass:
    """Test ExampleClass functionality."""
    
    @pytest.fixture
    def example_instance(self):
        """Create test instance."""
        return ExampleClass()
    
    def test_basic_functionality(self, example_instance):
        """Test basic operation."""
        result = example_instance.method()
        assert result is not None
        assert result == expected_value
    
    def test_edge_case(self, example_instance):
        """Test edge case handling."""
        result = example_instance.method(edge_input)
        assert result handles edge case
    
    @patch('src.example.external_dependency')
    def test_with_mock(self, mock_dep, example_instance):
        """Test with mocked dependency."""
        mock_dep.return_value = 'mocked'
        result = example_instance.method()
        assert result == 'expected'
        mock_dep.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Integration Test Example

```python
@pytest.mark.integration
class TestComponentIntegration:
    """Test component interactions."""
    
    def test_end_to_end_flow(self):
        """Test complete data flow."""
        # Setup components
        component_a = ComponentA()
        component_b = ComponentB()
        
        # Test interaction
        result = component_a.process()
        component_b.receive(result)
        
        # Verify results
        assert component_b.state == 'processed'
```

## Troubleshooting

### Common Issues

#### Issue: Import Errors
```
ModuleNotFoundError: No module named 'face_recognition'
```

**Solution**: Install missing dependencies
```bash
# Ubuntu/Debian
sudo apt-get install cmake libboost-all-dev
pip install face_recognition

# macOS
brew install cmake
pip install face_recognition
```

#### Issue: Slow Tests
```
Tests take too long to run
```

**Solution**: Run only unit tests
```bash
# Fast unit tests only (3-5 seconds)
python3 -m pytest tests/unit/ -v

# Skip slow tests
python3 -m pytest -m "not slow"
```

#### Issue: Coverage Not Updating
```
Coverage report shows old data
```

**Solution**: Clear cache
```bash
rm -rf .coverage htmlcov/ coverage.xml
python3 -m pytest --cov=src --cov=config
```

## Best Practices

### Writing Tests

1. **Descriptive Names**: `test_component_handles_edge_case`
2. **Docstrings**: Explain what is tested
3. **Arrange-Act-Assert**: Clear test structure
4. **Mock External**: Keep tests isolated
5. **Edge Cases**: Test boundary conditions
6. **Error Handling**: Test failure paths

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end workflow tests
├── performance/    # Benchmark tests
├── security/       # Security tests
├── fixtures/       # Test data
├── framework/      # Test utilities
└── conftest.py    # Shared fixtures
```

## Metrics

### Test Execution Time

| Test Suite | Time | Tests |
|-----------|------|-------|
| Unit | ~5s | 82 |
| Integration | ~30s | ~50 |
| E2E | ~2m | ~20 |
| Performance | ~1m | ~30 |
| Security | ~30s | ~25 |
| **Total** | **~4m** | **~210** |

### Coverage by Module Type

| Module Type | Files | Coverage |
|-------------|-------|----------|
| Config | 15 | ~15% |
| Pipeline | 6 | ~5% |
| Communication | 4 | ~40% |
| Storage | 9 | ~10% |
| Hardware | 5 | ~10% |
| Detectors | 4 | ~45% |
| **Total** | **~50** | **~25.5%** |

## Contributing

When adding new tests:

1. Follow existing patterns in `tests/unit/`
2. Run tests before committing
3. Update documentation if needed
4. Ensure 100% pass rate
5. Add to appropriate category

## Support

For help:
- Check this guide
- Review `TEST_EXECUTION_GUIDE.md`
- Examine existing test examples
- Consult CI/CD workflow results

---

**Last Updated**: Issue #25 completion
**Status**: Foundation complete, roadmap defined
**Next Phase**: Pipeline component tests
