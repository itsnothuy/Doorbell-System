# CI/CD Quick Reference Card

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-ci.txt

# 2. Validate environment
python scripts/testing/validate_test_environment.py

# 3. Run tests
pytest tests/ -v -m "not (hardware or gpu or load)"
```

## 📦 Requirements Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `requirements-ci.txt` | Fast, guaranteed CI deps | CI/CD pipelines |
| `requirements-testing.txt` | Full test suite deps | Local development |
| `requirements.txt` | Core application deps | Basic usage |
| `requirements-web.txt` | Web interface deps | Web development |

## 🧪 Test Commands

### Basic Testing
```bash
# All tests (excluding hardware/gpu/load)
pytest tests/ -v -m "not (hardware or gpu or load)"

# With coverage
pytest tests/ -v --cov=src --cov=config --cov-report=term-missing

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

### Test Markers
```bash
# By marker
pytest -v -m unit           # Unit tests only
pytest -v -m integration    # Integration only
pytest -v -m "not slow"     # Exclude slow tests

# Common exclusions
pytest -v -m "not (hardware or gpu or load or e2e)"
```

## 🔧 Common Fixes

### Import Errors
```bash
export PYTHONPATH=$PWD
pip install -r requirements-ci.txt
```

### Pre-commit Issues
```bash
pre-commit run --all-files
black src/ config/ tests/
ruff check --fix src/ config/ tests/
```

### Dependency Problems
```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements-ci.txt
pip install -r requirements-testing.txt
```

## 🏗️ Workflows

### Main Workflows
- **ci.yml** - Comprehensive CI/CD (multi-platform, multi-Python version)
- **comprehensive-tests.yml** - Full test suite with quality gates
- **codeql.yml** - Security scanning

### Trigger Branches
- Push: `master`, `main`, `develop`, `copilot/**`
- PR: `master`, `main`, `develop`

## ✅ Quality Gates

| Gate | Status | Required |
|------|--------|----------|
| Code Quality | Must Pass | ✅ Yes |
| Unit Tests | Should Pass | ⚠️ Warn |
| Integration | Optional | ❌ No |
| E2E Tests | Optional | ❌ No |
| Coverage | Target 70%+ | ⚠️ Warn |

## 🛠️ Utility Scripts

```bash
# Validate test environment
python scripts/testing/validate_test_environment.py

# Auto-add test markers
python scripts/testing/mark_tests.py

# Generate coverage report
python scripts/testing/generate_coverage_report.py

# Run full test suite
python scripts/testing/run_full_test_suite.py
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `export PYTHONPATH=$PWD` |
| Missing deps | `pip install -r requirements-ci.txt` |
| Pre-commit fails | `pre-commit run --all-files` |
| Coverage fails | `coverage run -m pytest tests/` |
| System deps | See docs/CI_CD_TESTING_GUIDE.md |

## 📊 Coverage

```bash
# Run with coverage
pytest --cov=src --cov=config --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🔐 Security Checks

```bash
# Bandit (security linting)
bandit -r src/ config/

# Safety (dependency vulnerabilities)
safety check

# Pre-commit security hooks
pre-commit run bandit --all-files
```

## 🎯 Best Practices

1. ✅ Test locally before pushing
2. ✅ Run pre-commit hooks
3. ✅ Use appropriate test markers
4. ✅ Check CI logs for warnings
5. ✅ Keep tests fast (< 15 min)
6. ✅ Update deps carefully

## 📖 Full Documentation

- [docs/CI_CD_TESTING_GUIDE.md](./CI_CD_TESTING_GUIDE.md) - Complete guide
- [README_TESTING.md](../README_TESTING.md) - Testing overview
- [TEST_EXECUTION_GUIDE.md](../TEST_EXECUTION_GUIDE.md) - Detailed execution

---

**Need help?** Check the full guide: [docs/CI_CD_TESTING_GUIDE.md](./CI_CD_TESTING_GUIDE.md)
