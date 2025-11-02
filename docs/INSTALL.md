# Installation Guide

Multiple installation options to fit your workflow!

## Option 1: Install Globally as a Tool ⭐ RECOMMENDED

Install once, use anywhere:

```bash
# Install from GitHub
uv tool install git+https://github.com/yourusername/claude-patterns

# Run from anywhere
claude-patterns
```

**Pros:**

- ✅ Single command installation
- ✅ Available globally in your PATH
- ✅ Auto-updates with `uv tool upgrade claude-patterns`
- ✅ Clean uninstall with `uv tool uninstall claude-patterns`

---

## Option 2: Run Directly with uv (No Installation)

Perfect for quick testing or one-time use:

```bash
# Clone the repo
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns

# Run with uv (auto-installs dependencies)
uv run --directory . python -m claude_patterns
```

**Pros:**

- ✅ No installation needed
- ✅ Dependencies auto-installed on first run
- ✅ Great for development/testing

---

## Option 3: One-Line Install Script

For users who want maximum convenience:

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/claude-patterns/main/install.sh | bash
```

This script:

1. Checks if `uv` is installed (installs if missing)
2. Installs `claude-patterns` as a tool
3. Verifies installation

---

## Option 4: Local Development Setup

For contributing or customizing:

```bash
# Clone the repository
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns

# Install in editable mode
uv tool install -e .

# Now changes to source code are immediately reflected
# Edit scripts and run claude-patterns
```

---

## Verifying Installation

After installation, verify it works:

```bash
claude-patterns --help
```

You should see:

```
usage: claude-patterns [-h] [--min-cluster-size MIN_CLUSTER_SIZE]
                       conversations_folder

Generate custom slash commands from Claude Code conversation history
```

---

## Updating

### Tool Installation

```bash
uv tool upgrade claude-patterns
```

### Development Installation

```bash
cd claude-patterns
git pull
uv sync
```

---

## Uninstalling

```bash
uv tool uninstall claude-patterns
```

---

## Troubleshooting

### uv not found

Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Permission denied

On macOS/Linux, make scripts executable:

```bash
chmod +x *.py
```

### Import errors

If using Option 2 (direct run), ensure you're in the project directory:

```bash
cd claude-patterns
uv run --directory . python -m claude_patterns
```

### Claude SDK authentication

Ensure you're logged into Claude Code:

```bash
claude
```

Or set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## Next Steps

After installation, see [README.md](README.md) for usage examples and configuration options.
