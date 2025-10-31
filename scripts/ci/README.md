# CI/CD Test Infrastructure Guide

This directory contains scripts for CI/CD test optimization and monitoring.

## Scripts

### `smart_test_selection.py`
Intelligently selects which tests to run based on changed files.

**Usage:**
```bash
# Get selected test files
python scripts/ci/smart_test_selection.py --base-branch master

# Save to file for CI
python scripts/ci/smart_test_selection.py --base-branch master --output-file selected-tests.txt

# Get pytest arguments format
python scripts/ci/smart_test_selection.py --format pytest-args
```

**Features:**
- Maps source files to their corresponding test files
- Always includes core system tests
- Handles multiple test patterns (unit, integration, etc.)
- Fallback to full suite if git is unavailable

### `performance_monitor.py`
Monitors test performance and detects regressions.

**Usage:**
```bash
# Check for regressions
python scripts/ci/performance_monitor.py \
  --junit-xml pytest-results.xml \
  --threshold 0.2

# Save new baseline
python scripts/ci/performance_monitor.py \
  --junit-xml pytest-results.xml \
  --save-baseline

# Get regression count (for scripting)
python scripts/ci/performance_monitor.py \
  --junit-xml pytest-results.xml \
  --count
```

**Features:**
- Compares against baseline performance metrics
- Configurable regression threshold (default 20%)
- Generates detailed regression reports
- Tracks test execution times

### `github_test_summary.py`
Generates comprehensive test summaries for GitHub Actions.

**Usage:**
```bash
# Generate summary for GitHub step summary
python scripts/ci/github_test_summary.py \
  --junit-xml pytest-results.xml \
  --coverage-json coverage.json \
  >> $GITHUB_STEP_SUMMARY

# Save to file
python scripts/ci/github_test_summary.py \
  --junit-xml pytest-results.xml \
  --coverage-json coverage.json \
  --output test-summary.md
```

**Features:**
- Parses JUnit XML test results
- Includes coverage statistics
- Shows failed tests with details
- Identifies slow tests
- Generates markdown with emojis for better readability

## Running Tests Locally

### Quick Test Run
```bash
# Run tests with default settings
pytest tests/

# Run with parallel execution (recommended)
pytest tests/ -n auto

# Run specific test categories
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "not (hardware or gpu)"
```

### Full Test Suite with Coverage
```bash
# Run all tests with coverage
pytest tests/ -v \
  -n auto \
  --cov=src --cov=config \
  --cov-report=html \
  --cov-report=xml \
  --junit-xml=pytest-results.xml
```

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Check for regressions
pytest tests/ --junit-xml=pytest-results.xml
python scripts/ci/performance_monitor.py --junit-xml pytest-results.xml
```

## CI/CD Integration

### GitHub Actions Workflow

The optimized test workflow (`optimized-tests.yml`) includes:

1. **Code Quality** (10 min)
   - Linting with ruff, black, isort
   - Fast feedback on code style

2. **Unit Tests** (15 min)
   - Smart test selection (only run affected tests)
   - Parallel execution with pytest-xdist
   - Automatic worker count optimization
   - Performance regression detection

3. **Integration Tests** (20 min)
   - Parallel execution with 2 workers
   - Isolated test environments
   - Comprehensive system testing

4. **Performance Tests** (15 min)
   - Benchmarking with pytest-benchmark
   - Baseline comparison
   - Only on main branch pushes

5. **Quality Gates** (5 min)
   - Aggregates all results
   - Enforces quality standards
   - Generates comprehensive summary

### Caching Strategy

The workflow uses multiple cache levels:

1. **Python packages** - `pip` cache via `actions/setup-python@v5`
2. **System dependencies** - APT cache for faster installs
3. **Pre-commit environments** - Speeds up linting
4. **Test results** - Baseline performance data

### Performance Optimizations

- **Parallel testing**: Uses pytest-xdist with auto worker count
- **Smart test selection**: Only runs tests affected by changes
- **Early termination**: `--maxfail=10` stops after 10 failures
- **Timeout protection**: 5-minute timeout per test
- **Matrix optimization**: Strategic exclusions to save time

## Test Infrastructure Features

### Test Isolation
- Per-worker temporary directories
- Process-safe database fixtures
- Automatic cleanup after tests
- Memory management with garbage collection

### Hardware Mocking
- Automatic hardware detection
- Fallback mocks for CI environments
- Consistent behavior across platforms
- No external dependencies in unit tests

### Network Isolation
- Mock external services (Telegram, etc.)
- No real network calls in unit tests
- Configurable mock responses
- Request/response validation

## Performance Metrics

### Target Times (Issue #27)
- **Total CI/CD**: < 30 minutes (from 45-60 minutes)
- **Unit tests**: < 3 minutes (from 8-12 minutes)
- **Integration tests**: < 8 minutes (from 15-20 minutes)
- **Cache hit rate**: > 80%
- **Parallel efficiency**: > 70% CPU utilization

### Monitoring
- Baseline performance tracking
- Automatic regression detection
- Test duration reporting
- Coverage trend analysis

## Troubleshooting

### Tests Failing Locally
```bash
# Check test isolation
pytest tests/ -vv --tb=short

# Run without parallelism for debugging
pytest tests/ -v

# Run specific test
pytest tests/test_specific.py::test_function -vv
```

### Performance Issues
```bash
# Profile test execution
pytest tests/ --durations=20

# Check for slow tests
pytest tests/ -v --durations=0 | grep "slow"

# Memory profiling
python -m memory_profiler pytest tests/
```

### Cache Issues
```bash
# Clear pytest cache
rm -rf .pytest_cache

# Clear coverage cache
rm -f .coverage coverage.xml

# Rebuild baseline
python scripts/ci/performance_monitor.py --junit-xml pytest-results.xml --save-baseline
```

## Contributing

When adding new tests:

1. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
2. **Keep tests fast**: Unit tests should run in < 1 second
3. **Mock external dependencies**: Never call real APIs in tests
4. **Test isolation**: Each test should be independent
5. **Update baselines**: If changing test structure significantly

## References

- [Issue #27: Test Infrastructure Hardening](https://github.com/itsnothuy/Doorbell-System/issues/27)
- [pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/)
- [GitHub Actions Cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
