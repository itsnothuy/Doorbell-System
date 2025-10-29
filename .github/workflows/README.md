# GitHub Actions Workflows

## Comprehensive Test Suite

The `comprehensive-tests.yml` workflow provides automated testing for the Doorbell Security System.

### Workflow Triggers

- **Push**: On main, develop, and copilot/** branches
- **Pull Request**: To main and develop branches
- **Schedule**: Nightly at 2 AM UTC
- **Manual**: Via workflow_dispatch

### Jobs

1. **Code Quality Checks**
   - Black formatting
   - Ruff linting
   - isort import sorting

2. **Unit Tests**
   - Tests across Python 3.10, 3.11, 3.12
   - Coverage reporting to Codecov

3. **Integration Tests**
   - Component interaction testing
   - Database integration testing

4. **End-to-End Tests**
   - Complete system scenarios
   - Doorbell to notification flows

5. **Performance Tests**
   - Throughput benchmarks
   - Latency measurements

6. **Security Tests**
   - Input validation
   - Vulnerability scanning
   - Bandit and Safety checks

7. **Coverage Report**
   - Comprehensive coverage analysis
   - HTML, XML, and JSON reports
   - Codecov integration

8. **Quality Gates**
   - Ensures all checks pass before merge

### Usage

The workflow runs automatically on push and PR. To run manually:

1. Go to Actions tab on GitHub
2. Select "Comprehensive Test Suite"
3. Click "Run workflow"

### Requirements

All dependencies are installed automatically by the workflow. See `pyproject.toml` for the complete list.
