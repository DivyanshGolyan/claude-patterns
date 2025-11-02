# Installation Guide

## Option 1: Install Globally as a Tool (Recommended)

Install once, use anywhere:

```bash
uv tool install git+https://github.com/yourusername/claude-patterns
claude-patterns
```

**Advantages:**

- Single command installation
- Available globally in your PATH
- Easy updates: `uv tool upgrade claude-patterns`
- Clean removal: `uv tool uninstall claude-patterns`

---

## Option 2: Run Directly with uv (No Installation)

For testing or one-time use:

```bash
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns
uv run --directory . python -m claude_patterns
```

**Advantages:**

- No installation required
- Dependencies automatically managed
- Useful for development and testing

---

## Option 3: One-Line Install Script

For automated installation:

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/claude-patterns/main/install.sh | bash
```

This script checks for `uv` (installing if needed), installs `claude-patterns`, and verifies the installation.

---

## Option 4: Local Development Setup

For contributing or customizing:

```bash
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns
uv tool install -e .
```

Changes to source code are immediately reflected when running `claude-patterns`.

---

## Verifying Installation

Test the installation:

```bash
claude-patterns --help
```

## Updating

**Tool installation:**

```bash
uv tool upgrade claude-patterns
```

**Development installation:**

```bash
cd claude-patterns && git pull && uv sync
```

## Uninstalling

```bash
uv tool uninstall claude-patterns
```

---

## Troubleshooting

**`uv not found`**

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Import errors**

Ensure you're in the project directory when using Option 2:

```bash
cd claude-patterns
uv run --directory . python -m claude_patterns
```

**Authentication errors**

Log into Claude Code or set an API key:

```bash
claude  # Login to Claude Code
# OR
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Next Steps

See [README.md](../README.md) for usage examples and configuration options.
