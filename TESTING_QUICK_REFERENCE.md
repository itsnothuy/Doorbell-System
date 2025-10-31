# Test Infrastructure Quick Reference Card

## 🚀 Quick Commands

### Development Testing
```bash
# Quick feedback (< 30s)
./scripts/testing/run_tests_local.sh quick

# Only changed tests (~1-2 min)
./scripts/testing/run_tests_local.sh fast

# Watch mode (continuous)
./scripts/testing/run_tests_local.sh watch

# Single test file
pytest tests/unit/test_myfeature.py -v
```

### Complete Testing
```bash
# All tests in parallel (~5-10 min)
./scripts/testing/run_tests_local.sh parallel

# With coverage report
./scripts/testing/run_tests_local.sh coverage

# Unit tests only
./scripts/testing/run_tests_local.sh unit
```

### Debugging
```bash
# Debug mode (verbose)
./scripts/testing/run_tests_local.sh debug

# Single test with debugger
pytest tests/unit/test_myfeature.py::test_case -vv --pdb

# Show local variables
pytest tests/ -l --tb=long
```

## 📊 CI/CD Scripts

### Smart Test Selection
```bash
# Get selected tests
python scripts/ci/smart_test_selection.py \
  --base-branch master

# Save to file
python scripts/ci/smart_test_selection.py \
  --output-file selected-tests.txt
```

### Performance Monitoring
```bash
# Check regressions (20% threshold)
python scripts/ci/performance_monitor.py \
  --junit-xml pytest-results.xml

# Save new baseline
python scripts/ci/performance_monitor.py \
  --junit-xml pytest-results.xml \
  --save-baseline
```

### Test Reporting
```bash
# Generate GitHub summary
python scripts/ci/github_test_summary.py \
  --junit-xml pytest-results.xml \
  --coverage-json coverage.json
```

## 🏷️ Test Markers

### Run Specific Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Exclude hardware tests
pytest -m "not (hardware or gpu)"
```

### Available Markers
- `unit` - Fast, isolated unit tests
- `integration` - Integration tests
- `e2e` - End-to-end tests
- `performance` - Performance benchmarks
- `slow` - Tests taking > 5 seconds
- `hardware` - Requires physical hardware
- `gpu` - Requires GPU
- `flaky` - Known flaky tests

## ⚡ Performance Options

### Parallel Execution
```bash
# Auto-detect workers
pytest tests/ -n auto

# Specific worker count
pytest tests/ -n 4 --dist worksteal
```

### Coverage Options
```bash
# HTML report
pytest --cov=src --cov-report=html

# Terminal + HTML
pytest --cov=src \
  --cov-report=term-missing \
  --cov-report=html

# All formats
pytest --cov=src \
  --cov-report=html \
  --cov-report=xml \
  --cov-report=json \
  --cov-report=term-missing
```

## 🔍 Troubleshooting

### Common Issues
```bash
# Clean test cache
rm -rf .pytest_cache htmlcov .coverage

# Verify environment
python scripts/testing/validate_test_environment.py

# Check test collection
pytest --collect-only

# Show slowest tests
pytest --durations=20

# Run test multiple times
pytest tests/test_flaky.py --count=10
```

### Debug Test Failures
```bash
# Stop on first failure
pytest tests/ -x

# Show full traceback
pytest tests/ --tb=long

# Show local variables
pytest tests/ -l --showlocals

# Enter debugger on failure
pytest tests/ --pdb
```

## 📁 Key Files

### Configuration
- `pyproject.toml` - Pytest configuration
- `tests/conftest.py` - Test fixtures
- `.github/workflows/optimized-tests.yml` - CI workflow

### Scripts
- `scripts/ci/smart_test_selection.py` - Test selection
- `scripts/ci/performance_monitor.py` - Performance tracking
- `scripts/ci/github_test_summary.py` - Report generation
- `scripts/testing/run_tests_local.sh` - Local runner

### Documentation
- `docs/TESTING_INFRASTRUCTURE.md` - Complete guide
- `scripts/ci/README.md` - CI scripts guide
- `TEST_INFRASTRUCTURE_SUMMARY.md` - Implementation summary

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| Total CI/CD | < 30 min |
| Unit Tests | < 3 min |
| Integration | < 8 min |
| Cache Hit Rate | > 80% |
| Parallel Efficiency | > 70% |
| Test Flakiness | < 2% |

## 🎯 Best Practices

### Writing Tests
1. ✅ Keep unit tests < 1 second
2. ✅ Test one behavior per test
3. ✅ Use appropriate markers
4. ✅ Mock external dependencies
5. ✅ Ensure test isolation
6. ✅ Use descriptive names

### Running Tests
1. ✅ Use quick/fast mode during development
2. ✅ Run full suite before pushing
3. ✅ Check coverage regularly
4. ✅ Fix flaky tests immediately
5. ✅ Monitor performance trends
6. ✅ Update baselines after improvements

## 🆘 Getting Help

1. Check documentation: `docs/TESTING_INFRASTRUCTURE.md`
2. View CI guide: `scripts/ci/README.md`
3. Check logs: `test-*.log` files
4. Run in debug mode
5. Check GitHub Actions logs

---

**Quick Links:**
- [Full Documentation](docs/TESTING_INFRASTRUCTURE.md)
- [CI Scripts Guide](scripts/ci/README.md)
- [Issue #27](https://github.com/itsnothuy/Doorbell-System/issues/27)

**Last Updated:** 2025-10-31
