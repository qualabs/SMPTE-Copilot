#!/bin/bash
# Script to run Ruff linter on the project

set -e

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo "Ruff is not installed. Installing from dev dependencies..."
    pip install -e ".[dev]"
fi

# Check for --fix flag
if [[ "$1" == "--fix" ]]; then
    echo "Running Ruff linter with auto-fix..."
    ruff check --fix --unsafe-fixes src/
    echo "✓ Fixed all auto-fixable issues!"
else
    # Run ruff check
    echo "Running Ruff linter..."
    ruff check src/
    echo ""
    echo "Tip: Run './lint.sh --fix' to automatically fix issues"
fi

# Optionally run ruff format check (uncomment if you want to enforce formatting)
# echo "Checking code formatting..."
# ruff format --check src/

echo "Linting complete!"

