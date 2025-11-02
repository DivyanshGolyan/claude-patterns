#!/bin/bash
#
# Installation script for claude-patterns
# Usage: curl -fsSL https://raw.githubusercontent.com/yourusername/claude-patterns/main/install.sh | bash
#

set -e

echo "================================"
echo "Claude Patterns Installer"
echo "================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv (ultra-fast Python package manager)..."

    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        # macOS/Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    # Source shell config to get uv in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    echo "uv installed successfully"
else
    echo "uv is already installed"
fi

echo ""
echo "Installing claude-patterns..."

# Install claude-patterns from GitHub
uv tool install git+https://github.com/yourusername/claude-patterns

echo ""
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  claude-patterns ~/path/to/conversations"
echo ""
echo "For more information, run:"
echo "  claude-patterns --help"
echo ""
echo "================================"
