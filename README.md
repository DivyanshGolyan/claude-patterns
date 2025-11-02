# claude-patterns

> Automatically generate reusable slash commands for Claude Code by analyzing your conversation history.

## Overview

Claude Code users often repeat similar requests across conversations. This tool uses machine learning to identify these patterns and automatically generates custom slash commands, saving you time and improving your workflow.

**What it does:**

- Analyzes your Claude Code conversation history
- Uses semantic clustering to group similar requests
- Generates slash commands for frequently-used patterns using the Claude Agent SDK
- Integrates directly into your `.claude/commands/` directory

## Installation

**Recommended: Install globally**

```bash
uv tool install git+https://github.com/yourusername/claude-patterns
```

**Alternative: Run from source**

```bash
git clone https://github.com/yourusername/claude-patterns
cd claude-patterns
uv run python -m claude_patterns
```

See [docs/INSTALL.md](docs/INSTALL.md) for additional installation options and troubleshooting.

## Features

- **Smart Pattern Detection**: Uses sentence-transformers for semantic similarity analysis
- **GPU Acceleration**: Automatically detects CUDA/MPS for faster processing
- **Duplicate Prevention**: Checks existing commands before generating new ones
- **High-Quality Output**: Enforces concise, actionable command instructions
- **Zero Configuration**: Auto-detects conversation directory from current project

## Usage

**Basic usage:**

```bash
claude-patterns
```

The tool will automatically:

1. Detect your current project's conversation directory
2. Extract and analyze conversation history
3. Identify recurring patterns using semantic clustering
4. Generate slash commands in `.claude/commands/`

**Common options:**

```bash
claude-patterns --min-cluster-size 3    # Require at least 3 similar messages per pattern
claude-patterns --threshold 0.6         # Adjust similarity threshold (lower = stricter)
```

**Available options:**

- `--min-cluster-size N`: Minimum messages per cluster (default: 2)
- `--threshold N`: Distance threshold for clustering (default: 0.7)
- `--model NAME`: Sentence-transformer model (default: all-MiniLM-L6-v2)
- `--max-message-length N`: Max characters per message (default: 500)
- `--max-messages N`: Max messages per cluster (default: 20)

## Authentication

The tool automatically uses your Claude Code subscription credentials when logged in - no separate API key required.

**Alternative:** Set `ANTHROPIC_API_KEY` environment variable to use API credits.

## Example Output

Generated slash commands appear in `.claude/commands/` with frontmatter:

```markdown
---
description: Verify and troubleshoot GitHub authentication
argument-hint: [optional: account-name]
---

Help me verify my GitHub authentication setup and resolve any connection issues...
```

**Using your generated commands:**

```bash
/verify-github-auth
```

## How It Works

The tool uses a three-stage pipeline:

1. **Extraction**: Reads conversation files from `~/.claude/projects/`, filtering out system-generated messages
2. **Clustering**: Uses sentence-transformers to compute semantic embeddings and identify similar requests
3. **Generation**: Uses the Claude Agent SDK to analyze clusters and generate appropriate slash commands

Each generated command is checked against existing commands to prevent duplicates.

## Project Structure

```
src/claude_patterns/
├── cli.py                # Main pipeline orchestration
├── extraction.py         # Message extraction from conversation files
├── clustering.py         # Semantic clustering with GPU support
├── generation.py         # Command generation via Claude SDK
└── prompts.py            # Prompt templates and filtering
```

Full documentation available in [`docs/`](docs/).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Active Claude Code subscription (or Anthropic API key)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Resources

- [Claude Code Documentation](https://docs.claude.com/en/docs/claude-code)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [Slash Commands Guide](https://docs.claude.com/en/docs/claude-code/slash-commands)

## License

This project is provided as-is for educational and personal use.
