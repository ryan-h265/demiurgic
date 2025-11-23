#!/bin/bash
# Test runner script for Demiurgic project

set -e

echo "==================================="
echo "Demiurgic Test Suite"
echo "==================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install test dependencies if needed
if ! python -c "import pytest" 2>/dev/null; then
    echo "Installing pytest..."
    pip install pytest pytest-cov
fi

echo ""
echo "Running tests..."
echo ""

# Run tests with different verbosity levels based on argument
if [ "$1" == "-v" ] || [ "$1" == "--verbose" ]; then
    pytest tests/ -v --tb=short
elif [ "$1" == "-vv" ]; then
    pytest tests/ -vv
elif [ "$1" == "--coverage" ]; then
    pytest tests/ -v --cov=src --cov-report=term-missing
elif [ "$1" == "--quick" ]; then
    # Run only fast tests (exclude slow ones)
    pytest tests/ -v -m "not slow"
else
    # Default: concise output
    pytest tests/ --tb=short
fi

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $exit_code)"
fi

exit $exit_code
