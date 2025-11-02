# Distribution Guide

This document explains how users can run `claude-patterns` with a **single command** from anywhere.

## 🎯 Best Options for End Users

### 1. Install as Global Tool (Best for Most Users)

```bash
uv tool install git+https://github.com/yourusername/claude-patterns
```

**Then run from anywhere:**

```bash
claude-patterns ~/conversations
```

**How it works:**

- `uv` downloads the repository
- Installs dependencies automatically
- Creates CLI commands in user's PATH
- Available globally on the system

**Pros:**

- ✅ Single command installation
- ✅ Works from any directory
- ✅ Easy updates: `uv tool upgrade claude-patterns`
- ✅ Clean removal: `uv tool uninstall claude-patterns`

---

### 2. One-Line Installer Script (Easiest for Non-Technical Users)

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/claude-patterns/main/install.sh | bash
```

**How it works:**

- Checks if `uv` is installed (installs if missing)
- Installs `claude-patterns` using method #1
- Verifies installation

**Pros:**

- ✅ Zero manual steps
- ✅ Handles all prerequisites
- ✅ Perfect for documentation/tutorials

---

### 3. Direct Execution (Best for Testing/Development)

```bash
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns
uv run --directory . python -m claude_patterns ~/conversations
```

**How it works:**

- Clone repository
- `uv run` reads dependencies from pyproject.toml
- Creates temporary environment
- Installs dependencies automatically
- Runs the module

**Pros:**

- ✅ No installation needed
- ✅ Great for trying before installing
- ✅ Perfect for development
- ✅ Dependencies auto-managed

---

## 📦 How Dependency Packaging Works

This project uses standard Python packaging with `pyproject.toml`:

The `pyproject.toml` defines:

```toml
[project]
name = "claude-patterns"
dependencies = [
    "anyio>=4.11.0",
    "claude-agent-sdk>=0.1.6",
    "numpy>=2.3.4",
    "scikit-learn>=1.7.2",
    "sentence-transformers>=5.1.2",
]

[project.scripts]
claude-patterns = "claude_patterns.cli:main"
```

When you run `uv tool install`, it:

1. Reads `pyproject.toml`
2. Installs all dependencies
3. Creates CLI executables
4. Adds them to PATH

**Benefits:**

- Global CLI commands
- Professional package distribution with src layout
- Easy updates and removal
- Standard Python packaging practices

---

## 🚀 Publishing Options

### Option A: GitHub Only (Current)

Users install with:

```bash
uv tool install git+https://github.com/yourusername/claude-patterns
```

**Pros:**

- ✅ No separate publishing step
- ✅ Always installs latest code
- ✅ Works immediately after push

**Cons:**

- ❌ Requires full GitHub URL
- ❌ No version pinning (always main branch)

---

### Option B: Publish to PyPI (Future)

Publish once:

```bash
uv build
uv publish
```

Users install with:

```bash
uv tool install claude-patterns
```

**Pros:**

- ✅ Short, memorable command
- ✅ Version pinning available
- ✅ Discoverable on PyPI
- ✅ Official distribution channel

**Cons:**

- ❌ Requires PyPI account
- ❌ Manual publishing step
- ❌ Version management overhead

---

## 🔄 Update Workflow

### For GitHub Installation

**Users update with:**

```bash
uv tool upgrade claude-patterns
```

This pulls latest code from main branch.

### For PyPI Installation

**You publish a new version:**

```bash
# Update version in pyproject.toml
uv build
uv publish
```

**Users update with:**

```bash
uv tool upgrade claude-patterns
```

---

## 🌐 Distribution Comparison

| Method              | Installation                      | Usage              | Updates    | Best For      |
| ------------------- | --------------------------------- | ------------------ | ---------- | ------------- |
| **uv tool install** | `uv tool install git+...`         | `claude-patterns`  | Auto       | Most users    |
| **One-line script** | `curl ... \| bash`                | `claude-patterns`  | Manual     | Documentation |
| **Direct run**      | `git clone ...`                   | `uv run script.py` | `git pull` | Development   |
| **PyPI** (future)   | `uv tool install claude-patterns` | `claude-patterns`  | Auto       | Production    |

---

## 💡 Recommendation

**For now:** Use GitHub installation (Option A)

- Simple URL: `uv tool install git+https://github.com/yourusername/claude-patterns`
- No PyPI account needed
- Immediate availability after push

**For future:** Consider PyPI publication when:

- You want a short install command
- You need version stability
- The project is widely used

---

## 📝 Summary

Your users can run `claude-patterns` from anywhere with just **one command**:

```bash
# Option 1: Install globally
uv tool install git+https://github.com/yourusername/claude-patterns
claude-patterns ~/conversations

# Option 2: One-line installer
curl -fsSL https://raw.githubusercontent.com/yourusername/claude-patterns/main/install.sh | bash
claude-patterns ~/conversations

# Option 3: Direct run (no install)
git clone https://github.com/yourusername/claude-patterns && cd claude-patterns
uv run --directory . python -m claude_patterns ~/conversations
```

All three methods handle dependencies automatically. Choose based on your users' technical level and use case.
