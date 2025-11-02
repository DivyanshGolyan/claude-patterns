# Claude Code Slash Command Generator

Automatically generate custom slash commands for Claude Code by analyzing your conversation history patterns.

## Quick Start

### Option 1: Install Globally (Recommended)

Install once, use anywhere:

```bash
uv tool install git+https://github.com/yourusername/claude-patterns
claude-patterns
```

### Option 2: One-Line Install Script

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/claude-patterns/main/install.sh | bash
```

### Option 3: Run Without Installing

```bash
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns
uv run --directory . python -m claude_patterns
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed installation options and troubleshooting.

## How It Works

### Conversation Storage

Claude Code stores all conversations locally in:

```
~/.claude/projects/<encoded-path>/[session-uuid].jsonl
```

Where `<encoded-path>` is your project's absolute path with `/` replaced by `-`:

- Project: `/Users/name/Documents/GitHub/myproject`
- Storage: `~/.claude/projects/-Users-name-Documents-GitHub-myproject/`

### Dependency Management

This project uses standard Python packaging with `pyproject.toml`. When you install with `uv tool install`, it:

1. Reads dependencies from `pyproject.toml`
2. Creates an isolated virtual environment
3. Installs required packages automatically
4. Creates CLI entry points in your PATH

**Benefits:**

- Professional package structure with src layout
- All dependencies declared in one place
- Easy to install globally with `uv tool install`
- Automatic CLI command creation
- Standard Python packaging practices

## Usage

### Complete Pipeline (Recommended)

**If installed as a tool:**

```bash
claude-patterns --min-cluster-size 3
```

**If running directly:**

```bash
uv run python -m claude_patterns --min-cluster-size 3
```

**Options:**

- `--min-cluster-size N`: Minimum messages per cluster (default: 2)
- `--threshold N`: Distance threshold for clustering (default: 0.7, lower = stricter)
- `--model NAME`: Sentence-transformer model (default: all-MiniLM-L6-v2)
- `--max-message-length N`: Maximum characters per message sent to agent (default: 500)
- `--max-messages N`: Maximum messages per cluster sent to agent (default: 20)

## Authentication

The Claude Agent SDK automatically uses your **Claude Code subscription credentials** if you're logged in. No separate API key needed!

Alternative: Set `ANTHROPIC_API_KEY` environment variable to use API credits instead.

## What Gets Generated

The pipeline analyzes your conversation history and creates slash commands for common patterns:

**Example generated command:**

```markdown
---
description: Verify and troubleshoot GitHub authentication
argument-hint: [optional: account-name]
---

Help me verify my GitHub authentication setup...
```

**Usage in Claude Code:**

```bash
/verify-github-auth
```

## Advanced Options

### Clustering Parameters

```bash
uv run python -m claude_patterns.clustering user_messages.json \
    --threshold 0.7 \              # Distance threshold (lower = more similar)
    --min-size 3 \                 # Minimum messages per cluster
    --model all-MiniLM-L6-v2 \     # Sentence transformer model
    --limit 100                    # Process only first N messages (testing)
```

### Message Filtering

By default, the pipeline **automatically filters out** system-generated messages including:

- Bash commands and output
- Slash command invocations
- System interruptions
- Session continuation messages

This ensures cleaner clustering focused on actual user requests.

If you need to include system messages when using the extraction module directly:

```bash
# Extract without filtering (includes system messages)
uv run python -m claude_patterns.extraction
# Default behavior excludes system messages with --exclude-system
```

## Project Structure

```
claude-patterns/
├── src/
│   └── claude_patterns/
│       ├── __init__.py           # Package initialization
│       ├── __main__.py           # CLI entry point
│       ├── cli.py                # Main pipeline orchestration
│       ├── extraction.py         # Extract messages from JSONL files
│       ├── clustering.py         # Cluster similar messages
│       ├── generation.py         # Generate slash commands using Claude SDK
│       └── prompts.py            # Shared prompt templates
├── tests/
│   └── test_*.py                 # Test scripts
├── docs/
│   ├── INSTALL.md                # Installation guide
│   ├── DISTRIBUTION.md           # Distribution options
│   ├── PUBLISHING.md             # Publishing guide
│   └── SETUP.md                  # Setup instructions
├── README.md                     # This file
├── pyproject.toml                # Package configuration
└── .claude/commands/             # Generated slash commands (created when run)
```

## Architecture

The project follows Python best practices with a **src layout**:

- **Modular design**: Each component (extraction, clustering, generation) is a separate module
- **Direct imports**: The CLI orchestrator imports and calls functions directly (no subprocess overhead)
- **Reusable components**: Each module can be used independently or as part of the pipeline
- **Type hints**: Full type annotations for better IDE support and code quality
- **Proper packaging**: Standard setuptools configuration for easy installation

## Learn More

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 723 - Inline Script Metadata](https://peps.python.org/pep-0723/)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [Claude Code Slash Commands](https://docs.claude.com/en/docs/claude-code/slash-commands)

## Contributing

This project demonstrates modern Python packaging with `uv`. Feel free to use this pattern in your own projects!

## License

This project is provided as-is for educational and personal use.
