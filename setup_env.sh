#!/bin/bash
# Setup script for Demiurgic virtual environment
#
# Usage:
#   ./setup_env.sh           # Quick setup (core only)
#   ./setup_env.sh --full    # Full setup (with training deps)

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         Demiurgic Virtual Environment Setup              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" = "3.10" ]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.10 or higher is required"
    echo "   Please install Python 3.10+ and try again"
    exit 1
fi

echo "✓ Found Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# Create virtual environment
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at ./$VENV_DIR"
    read -p "   Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        rm -rf "$VENV_DIR"
    else
        echo "   Skipping environment creation"
        echo ""
        echo "To activate existing environment, run:"
        echo "   source $VENV_DIR/bin/activate"
        exit 0
    fi
fi

echo "Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_DIR"
echo "✓ Virtual environment created at ./$VENV_DIR"
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install dependencies
if [ "$1" == "--full" ]; then
    echo "Installing FULL dependencies (this may take a while)..."
    echo ""
    pip install -r requirements.txt
    echo ""
    echo "✓ All dependencies installed"
else
    echo "Installing CORE dependencies (minimal setup)..."
    echo ""
    pip install -r requirements-core.txt
    echo ""
    echo "✓ Core dependencies installed"
    echo ""
    echo "ℹ️  Note: To install training dependencies later, run:"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements-training.txt"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! ✓                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "To activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "To test the model:"
echo "   source venv/bin/activate"
echo "   python scripts/test_model_basic.py"
echo ""
echo "To deactivate when done:"
echo "   deactivate"
echo ""
