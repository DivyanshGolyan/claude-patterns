# Publishing to GitHub

Quick guide to make `claude-patterns` available for single-command installation.

## 📤 Steps to Publish

### 1. Update Repository URLs

Replace placeholders in these files:

**pyproject.toml:**

```toml
[project.urls]
Homepage = "https://github.com/YOURUSERNAME/claude-patterns"
Repository = "https://github.com/YOURUSERNAME/claude-patterns"
Issues = "https://github.com/YOURUSERNAME/claude-patterns/issues"
```

**install.sh:**

```bash
uv tool install git+https://github.com/YOURUSERNAME/claude-patterns
```

**README.md and INSTALL.md:**
Update all GitHub URLs with your actual username.

### 2. Update Author Info

**pyproject.toml:**

```toml
authors = [{ name = "Your Name", email = "[email protected]" }]
```

**LICENSE:**
Update copyright line with your name.

### 3. Create GitHub Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Claude Code slash command generator"

# Create repository on GitHub (via web UI or gh CLI)
gh repo create claude-patterns --public --source=. --remote=origin

# Push to GitHub
git push -u origin main
```

### 4. Test Installation

Verify users can install with:

```bash
uv tool install git+https://github.com/YOURUSERNAME/claude-patterns
```

### 5. Update Documentation

Create a GitHub README by adding badges and examples:

```markdown
# Claude Code Slash Command Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[Rest of README content...]
```

## 🏷️ Versioning (Optional but Recommended)

### Create a Release Tag

```bash
# Update version in pyproject.toml
# e.g., version = "0.1.0" -> "0.2.0"

git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git tag v0.2.0
git push origin main --tags
```

### Users Can Install Specific Versions

```bash
# Install latest
uv tool install git+https://github.com/YOURUSERNAME/claude-patterns

# Install specific version
uv tool install git+https://github.com/YOURUSERNAME/[email protected]
```

## 📝 Create a GitHub Release

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Choose tag: `v0.1.0`
4. Release title: `v0.1.0 - Initial Release`
5. Description: List features and changes
6. Publish release

## 🔄 Update Workflow

When you make changes:

```bash
# Make your changes
git add .
git commit -m "Add feature: XYZ"
git push origin main
```

Users update with:

```bash
uv tool upgrade claude-patterns
```

## 🌟 Optional Enhancements

### Add GitHub Actions CI/CD

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Run tests
        run: |
          uv run --version
```

### Add Documentation

Create `docs/` folder with:

- Usage examples
- API documentation
- Troubleshooting guide

Host on GitHub Pages for easy access.

### Add Issue Templates

Create `.github/ISSUE_TEMPLATE/` with templates for:

- Bug reports
- Feature requests
- Questions

## 📦 Alternative: Publish to PyPI

If you want users to install with just `uv tool install claude-patterns`:

```bash
# 1. Get PyPI account at https://pypi.org

# 2. Build the package
uv build

# 3. Publish (will prompt for credentials)
uv publish

# 4. Users can now install with short command
uv tool install claude-patterns
```

**Pros:**

- Shorter install command
- Official Python package
- Discoverable on PyPI

**Cons:**

- Need PyPI account
- Manual publishing process
- Version management required

## ✅ Verification Checklist

Before sharing publicly:

- [ ] All `YOURUSERNAME` placeholders replaced
- [ ] Author info updated in pyproject.toml and LICENSE
- [ ] Repository created on GitHub
- [ ] Code pushed to main branch
- [ ] Installation tested: `uv tool install git+https://github.com/...`
- [ ] CLI commands work: `claude-patterns --help`
- [ ] README has correct GitHub URLs
- [ ] install.sh has correct GitHub URL
- [ ] License file updated with your name

## 🎉 Ready to Share

Once published, users can install with:

```bash
uv tool install git+https://github.com/YOURUSERNAME/claude-patterns
```

Or use your one-line installer:

```bash
curl -fsSL https://raw.githubusercontent.com/YOURUSERNAME/claude-patterns/main/install.sh | bash
```

Share the GitHub URL and watch the stars roll in! ⭐
