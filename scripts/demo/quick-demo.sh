#!/bin/bash
# Quick Demo Runner
# Provides easy access to various demo modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section header
print_header() {
    echo ""
    print_color "$BLUE" "=========================================="
    print_color "$BLUE" "$1"
    print_color "$BLUE" "=========================================="
    echo ""
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_color "$RED" "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Main menu
print_header "Doorbell Security System - Demo Runner"

echo "Select demo mode:"
echo ""
echo "  1) Quick Demo (5 minutes) - Overview of key features"
echo "  2) Complete Demo (25 minutes) - Full system demonstration"
echo "  3) Setup Flow Only - Initial configuration walkthrough"
echo "  4) Daily Operations - Typical usage scenarios"
echo "  5) Advanced Features - AI analysis and multi-camera"
echo "  6) Administration - Monitoring and maintenance"
echo "  7) Troubleshooting - Diagnostics and support"
echo "  8) Interactive Mode - Step-by-step with user input"
echo "  9) Run All Tests - Validate demo system"
echo "  0) Exit"
echo ""

read -p "Enter your choice (0-9): " choice

case $choice in
    1)
        print_header "Running Quick Demo"
        python3 -m demo.orchestrator --quick
        ;;
    2)
        print_header "Running Complete Demo"
        python3 -m demo.orchestrator
        ;;
    3)
        print_header "Running Setup Flow Demo"
        python3 -m demo.flows.initial_setup
        ;;
    4)
        print_header "Running Daily Operations Demo"
        python3 -m demo.flows.daily_operation
        ;;
    5)
        print_header "Running Advanced Features Demo"
        python3 -m demo.flows.advanced_features
        ;;
    6)
        print_header "Running Administration Demo"
        python3 -m demo.flows.administration
        ;;
    7)
        print_header "Running Troubleshooting Demo"
        python3 -m demo.flows.troubleshooting
        ;;
    8)
        print_header "Running Interactive Demo"
        python3 -m demo.orchestrator --interactive
        ;;
    9)
        print_header "Running Demo Tests"
        if command -v pytest &> /dev/null; then
            pytest tests/demo/ -v
        else
            print_color "$YELLOW" "pytest not installed. Running basic tests..."
            python3 -m tests.demo.test_demo_flows
        fi
        ;;
    0)
        print_color "$GREEN" "Goodbye!"
        exit 0
        ;;
    *)
        print_color "$RED" "Invalid choice. Please run again and select 0-9."
        exit 1
        ;;
esac

# Print completion message
echo ""
print_color "$GREEN" "Demo completed successfully!"
echo ""
print_color "$BLUE" "Next steps:"
echo "  • Review documentation: docs/demo/"
echo "  • Try other demo modes"
echo "  • Deploy to production: ./setup.sh"
echo "  • Join community: https://github.com/itsnothuy/Doorbell-System"
echo ""
