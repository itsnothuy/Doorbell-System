# Example: GitHub Actions Integration with Test Orchestrator

This document provides example GitHub Actions workflow configurations for integrating the test orchestrator into your CI/CD pipeline.

## Basic Integration

```yaml
name: Test with Orchestrator

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing,monitoring]'
    
    - name: Run Test Orchestrator
      run: |
        python tests/run_orchestrator.py \
          --suites all \
          --workers 4 \
          --environment ci
    
    - name: Upload Test Reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: test-reports
        path: test-results/
```

## Fast Feedback for PRs

```yaml
name: PR Tests

on:
  pull_request:
    branches: [ main, develop ]

jobs:
  quick-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing]'
    
    - name: Run Fast Tests
      run: |
        python tests/run_orchestrator.py \
          --suites unit integration \
          --workers 4 \
          --fail-fast \
          --environment ci
    
    - name: Upload Reports
      uses: actions/upload-artifact@v4
      if: failure()
      with:
        name: test-reports
        path: test-results/
```

## Comprehensive Nightly Tests

```yaml
name: Nightly Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  comprehensive-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing,monitoring]'
    
    - name: Run Comprehensive Tests
      run: |
        python tests/run_orchestrator.py \
          --suites all \
          --workers 8 \
          --environment production_like \
          --output-dir nightly-results
    
    - name: Upload Test Reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: nightly-test-reports
        path: nightly-results/
    
    - name: Upload Coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        files: nightly-results/coverage_unit.json,nightly-results/coverage_integration.json
        flags: nightly
```

## Performance Regression Testing

```yaml
name: Performance Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing,monitoring]'
    
    - name: Restore Performance Baselines
      uses: actions/cache@v3
      with:
        path: tests/baselines/
        key: performance-baselines-${{ github.sha }}
        restore-keys: |
          performance-baselines-
    
    - name: Run Performance Tests
      run: |
        python tests/run_orchestrator.py \
          --suites performance \
          --workers 4 \
          --no-coverage
    
    - name: Check for Regressions
      run: |
        # Parse performance report and check for regressions
        python -c "
        import json
        from pathlib import Path
        report = json.loads(Path('test-results/performance_report.json').read_text())
        print('Performance metrics:', report)
        "
    
    - name: Upload Performance Reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: performance-reports
        path: test-results/
```

## Matrix Testing (Multiple Python Versions)

```yaml
name: Matrix Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing]'
    
    - name: Run Tests
      run: |
        python tests/run_orchestrator.py \
          --suites unit integration \
          --workers 2 \
          --environment ci
    
    - name: Upload Reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: test-reports-${{ matrix.os }}-py${{ matrix.python-version }}
        path: test-results/
```

## Security Testing

```yaml
name: Security Tests

on:
  push:
    branches: [ main, develop ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing]'
        pip install bandit safety
    
    - name: Run Security Tests
      run: |
        python tests/run_orchestrator.py \
          --suites security \
          --workers 2 \
          --no-coverage
    
    - name: Run Bandit Security Scan
      run: |
        bandit -r src/ -f json -o test-results/bandit-report.json || true
    
    - name: Upload Security Reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: security-reports
        path: test-results/
```

## Using Test Results in Status Checks

```yaml
name: Test Status Check

on:
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[dev,testing]'
    
    - name: Run Tests
      id: test_run
      run: |
        python tests/run_orchestrator.py \
          --suites unit integration \
          --workers 4 \
          --environment ci
    
    - name: Parse Test Results
      if: always()
      run: |
        python -c "
        import json
        from pathlib import Path
        
        report = json.loads(Path('test-results/test_report.json').read_text())
        
        print(f\"Status: {report['overall_status']}\")
        print(f\"Duration: {report['total_duration']:.2f}s\")
        
        for suite in report['suite_results']:
            print(f\"{suite['suite'].upper()}: {suite['passed']}/{suite['total_tests']} passed\")
        
        # Fail if tests failed
        if report['overall_status'] != 'passed':
            exit(1)
        "
    
    - name: Comment on PR
      if: always()
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const report = JSON.parse(fs.readFileSync('test-results/test_report.json', 'utf8'));
          
          let comment = `## Test Results\n\n`;
          comment += `**Status:** ${report.overall_status === 'passed' ? '✅' : '❌'} ${report.overall_status.toUpperCase()}\n`;
          comment += `**Duration:** ${report.total_duration.toFixed(2)}s\n\n`;
          comment += `### Suite Results\n\n`;
          
          for (const suite of report.suite_results) {
            comment += `- **${suite.suite.toUpperCase()}**: ${suite.passed}/${suite.total_tests} passed\n`;
          }
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
    
    - name: Upload Reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: test-reports
        path: test-results/
```

## Best Practices

1. **Use fail-fast for PR checks**: Get quick feedback on failures
2. **Parallel execution**: Use `--workers` to speed up tests
3. **Upload artifacts**: Always upload test reports for debugging
4. **Cache dependencies**: Use pip caching to speed up workflow
5. **Separate concerns**: Use different workflows for different test types
6. **Performance baselines**: Cache and track performance baselines over time
7. **Matrix testing**: Test across Python versions and OS platforms
8. **Security scans**: Run security tests regularly in scheduled workflows
