# Test Baselines

This directory contains baseline data for test performance monitoring and regression detection.

## Files

### `performance.json`
Contains baseline execution times for all tests. Used by the performance monitor to detect regressions.

**Structure:**
```json
{
  "test.module.TestClass.test_method": 0.123,
  "test.another.test_function": 0.045
}
```

**Usage:**
```bash
# Establish new baseline
python scripts/ci/performance_monitor.py --junit-xml pytest-results.xml --save-baseline

# Check for regressions
python scripts/ci/performance_monitor.py --junit-xml pytest-results.xml --threshold 0.2
```

## Updating Baselines

Baselines should be updated when:
1. Performance improvements are made (and verified)
2. Test structure changes significantly
3. Hardware/environment changes affect all tests consistently

**Never** update baselines to hide legitimate performance regressions!

## Baseline Management

- Baselines are committed to the repository
- CI/CD checks against these baselines automatically
- Updates require code review and justification
- Separate baselines may be needed for different environments (if needed)
