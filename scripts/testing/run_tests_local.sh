#!/usr/bin/env bash
#
# Run Tests Script
#
# Convenient wrapper for running tests with various configurations.
# This script provides preset configurations for common testing scenarios.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if dependencies are installed
check_dependencies() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.10+"
        exit 1
    fi
    
    if ! python -m pytest --version &> /dev/null; then
        print_error "pytest not installed. Run: pip install -r requirements-ci.txt"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [option]

Options:
    quick       Run quick tests (unit tests only, no coverage)
    unit        Run unit tests with coverage
    integration Run integration tests
    all         Run all tests (unit + integration + e2e)
    parallel    Run tests in parallel (requires pytest-xdist)
    fast        Run tests with smart selection (only changed)
    performance Run performance benchmarks
    coverage    Run tests and generate coverage report
    watch       Run tests in watch mode (re-run on changes)
    debug       Run tests in debug mode (verbose, no parallel)
    help        Show this help message

Examples:
    $0 quick                # Quick feedback loop during development
    $0 unit                 # Full unit test suite with coverage
    $0 parallel             # Fast parallel execution
    $0 fast                 # Only run tests for changed files

EOF
}

# Function to run quick tests
run_quick_tests() {
    print_info "Running quick tests (unit tests, no coverage)..."
    python -m pytest tests/unit/ -v -x \
        --maxfail=5 \
        -m "not (slow or hardware or gpu)" || {
        print_error "Quick tests failed"
        exit 1
    }
    print_success "Quick tests passed!"
}

# Function to run unit tests
run_unit_tests() {
    print_info "Running unit tests with coverage..."
    python -m pytest tests/ -v \
        --cov=src --cov=config \
        --cov-report=term-missing \
        --cov-report=html \
        -m "not (integration or e2e or hardware or gpu or load)" || {
        print_error "Unit tests failed"
        exit 1
    }
    print_success "Unit tests passed!"
    print_info "Coverage report: htmlcov/index.html"
}

# Function to run integration tests
run_integration_tests() {
    print_info "Running integration tests..."
    python -m pytest tests/integration/ -v \
        --cov=src --cov=config \
        --cov-report=term-missing \
        --maxfail=5 || {
        print_error "Integration tests failed"
        exit 1
    }
    print_success "Integration tests passed!"
}

# Function to run all tests
run_all_tests() {
    print_info "Running all tests (this may take a while)..."
    python -m pytest tests/ -v \
        --cov=src --cov=config \
        --cov-report=term-missing \
        --cov-report=html \
        -m "not (hardware or gpu)" || {
        print_error "Tests failed"
        exit 1
    }
    print_success "All tests passed!"
    print_info "Coverage report: htmlcov/index.html"
}

# Function to run tests in parallel
run_parallel_tests() {
    if ! python -c "import xdist" 2>/dev/null; then
        print_warning "pytest-xdist not installed. Installing..."
        pip install pytest-xdist
    fi
    
    print_info "Running tests in parallel..."
    CPU_COUNT=$(python -c "import os; print(min(os.cpu_count() or 1, 4))")
    print_info "Using $CPU_COUNT workers"
    
    python -m pytest tests/ -v \
        -n "$CPU_COUNT" \
        --dist worksteal \
        --cov=src --cov=config \
        --cov-report=term-missing \
        -m "not (hardware or gpu or load)" || {
        print_error "Parallel tests failed"
        exit 1
    }
    print_success "Parallel tests passed!"
}

# Function to run smart test selection
run_fast_tests() {
    print_info "Running smart test selection..."
    
    if ! python scripts/ci/smart_test_selection.py --output-file selected-tests.txt; then
        print_warning "Smart selection unavailable, running all tests"
        run_unit_tests
        return
    fi
    
    if [ ! -s selected-tests.txt ]; then
        print_info "No specific tests selected, running full suite"
        run_unit_tests
        return
    fi
    
    print_info "Selected tests:"
    cat selected-tests.txt
    
    TEST_FILES=$(cat selected-tests.txt | tr '\n' ' ')
    python -m pytest $TEST_FILES -v \
        --cov=src --cov=config \
        --cov-report=term-missing || {
        print_error "Fast tests failed"
        exit 1
    }
    print_success "Fast tests passed!"
}

# Function to run performance tests
run_performance_tests() {
    if ! python -c "import pytest_benchmark" 2>/dev/null; then
        print_warning "pytest-benchmark not installed. Installing..."
        pip install pytest-benchmark
    fi
    
    print_info "Running performance benchmarks..."
    python -m pytest tests/performance/ -v \
        --benchmark-only \
        --benchmark-json=benchmark-results.json || {
        print_warning "Performance tests completed with warnings"
    }
    print_success "Performance benchmarks completed!"
    
    if [ -f benchmark-results.json ]; then
        print_info "Benchmark results saved to benchmark-results.json"
    fi
}

# Function to generate coverage report
run_coverage_report() {
    print_info "Running tests and generating coverage report..."
    python -m pytest tests/ -v \
        --cov=src --cov=config \
        --cov-report=html \
        --cov-report=xml \
        --cov-report=json \
        --cov-report=term-missing \
        -m "not (hardware or gpu or load)" || {
        print_warning "Some tests failed, but coverage was generated"
    }
    
    print_success "Coverage report generated!"
    print_info "HTML report: htmlcov/index.html"
    print_info "XML report: coverage.xml"
    print_info "JSON report: coverage.json"
    
    # Run coverage report script if available
    if [ -f scripts/testing/generate_coverage_report.py ]; then
        python scripts/testing/generate_coverage_report.py --no-run --badge --markdown
    fi
}

# Function to run tests in watch mode
run_watch_tests() {
    if ! command -v pytest-watch &> /dev/null; then
        print_warning "pytest-watch not installed. Installing..."
        pip install pytest-watch
    fi
    
    print_info "Running tests in watch mode (press Ctrl+C to stop)..."
    print_info "Tests will re-run when files change"
    
    ptw tests/ -- -v --maxfail=5 -m "not (hardware or gpu or load or slow)"
}

# Function to run tests in debug mode
run_debug_tests() {
    print_info "Running tests in debug mode (verbose, no parallel)..."
    python -m pytest tests/ -vv \
        --tb=long \
        --showlocals \
        --maxfail=1 \
        -m "not (hardware or gpu or load)" || {
        print_error "Debug tests failed"
        exit 1
    }
}

# Main script logic
main() {
    check_dependencies
    
    case "${1:-help}" in
        quick)
            run_quick_tests
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        all)
            run_all_tests
            ;;
        parallel)
            run_parallel_tests
            ;;
        fast)
            run_fast_tests
            ;;
        performance)
            run_performance_tests
            ;;
        coverage)
            run_coverage_report
            ;;
        watch)
            run_watch_tests
            ;;
        debug)
            run_debug_tests
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
