#!/bin/bash

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
    exit_code=0
else
    # Run ruff check
    echo "Running Ruff linter..."
    ruff check src/ --show-files
    exit_code=$?
    echo ""
    if [ $exit_code -ne 0 ]; then
        echo "Tip: Run './scripts/lint.sh --fix' to automatically fix issues"
    fi
fi

echo "Linting complete!"
exit $exit_code