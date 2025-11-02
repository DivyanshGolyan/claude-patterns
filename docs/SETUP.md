# Setup Guide for Claude Patterns

## Quick Start

### 1. Install Dependencies with UV

```bash
# UV is already installed, initialize and sync
uv sync
```

### 2. Verify Authentication

**Good news!** If you're already logged in to Claude Code with a Pro/Max subscription, the SDK will automatically use your credentials. **No separate API key needed!**

#### Check Your Authentication

```bash
# Test if SDK can connect
uv run python tests/test_sdk_auth.py
```

If you see "✓ Success!", you're all set!

#### Alternative: Use an API Key (Optional)

If you prefer to use API credits instead of your subscription, or if you're not logged in to Claude Code:

1. Visit: https://console.anthropic.com/settings/keys
2. Click "Create Key"
3. Copy the key (starts with `sk-ant-`)
4. Set the environment variable:

```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'

# Or add permanently to ~/.zshrc:
echo "export ANTHROPIC_API_KEY='sk-ant-your-key-here'" >> ~/.zshrc
source ~/.zshrc
```

**Note:** API key usage bills separately from your Claude subscription.

## Usage

### Run the Complete Pipeline

The pipeline automatically extracts messages, clusters them, and generates slash commands in one step:

```bash
# If installed as a tool
claude-patterns

# Or run directly from the repository
uv run python -m claude_patterns
```

This will:

1. Extract all user messages from Claude Code JSONL files
2. Cluster messages by semantic similarity
3. Use Claude Agent SDK to generate relevant slash commands
4. Save them to `.claude/commands/` directory

**Options:**

```bash
claude-patterns \
  --min-cluster-size 2 \           # Minimum messages per cluster
  --threshold 0.7 \                # Distance threshold (lower = stricter)
  --model all-MiniLM-L6-v2 \       # Sentence-transformer model
  --max-message-length 500 \       # Max characters per message to agent
  --max-messages 20 \              # Max messages per cluster to agent
  --min-absolute 5 \               # Minimum absolute message count per cluster
  --min-percentage 0.03            # Minimum percentage of total messages (3%)
```

## Troubleshooting

### "No authentication credentials found" Error

The SDK checks for credentials in this order:

1. **Claude Code CLI** (preferred - uses your subscription)
2. **ANTHROPIC_API_KEY** environment variable
3. **CLAUDE_CODE_OAUTH_TOKEN** environment variable

**Solution 1: Make sure you're logged in to Claude Code**

```bash
claude
# If not logged in, follow the prompts
```

**Solution 2: Set an API key**

```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
```

### Authentication Options Comparison

| Feature            | Claude Code CLI (Default)     | API Key (Optional)          |
| ------------------ | ----------------------------- | --------------------------- |
| **Authentication** | Auto-detected from Claude CLI | `ANTHROPIC_API_KEY` env var |
| **Subscription**   | Uses Pro/Max plan             | Pay-per-use                 |
| **Setup**          | Log in once with `claude`     | Set environment variable    |
| **Key Format**     | Automatic                     | `sk-ant-...`                |
| **Billing**        | Included in subscription      | Separate API billing        |
| **Where to Get**   | Already have it!              | console.anthropic.com       |

### Project Path Detection

The generation module saves slash commands to `.claude/commands/` in your current working directory:

- Running from: `/Users/name/Documents/GitHub/my-project`
- Saves commands to: `/Users/name/Documents/GitHub/my-project/.claude/commands/`

## Project Structure

```
claude-patterns/
├── src/
│   └── claude_patterns/
│       ├── __init__.py          # Package initialization
│       ├── __main__.py          # CLI entry point
│       ├── cli.py               # Main pipeline orchestration
│       ├── extraction.py        # Extract messages from JSONL files
│       ├── clustering.py        # Cluster messages by similarity
│       ├── generation.py        # Generate slash commands using Claude SDK
│       └── prompts.py           # Shared prompt templates
├── tests/
│   ├── test_sdk_auth.py         # Test SDK authentication
│   └── test_sdk_response.py     # Test SDK response streaming
├── docs/                         # Documentation
├── pyproject.toml               # Package configuration
├── .venv/                       # Virtual environment (auto-created)
└── clusters_*/                  # Generated cluster folders (timestamped)
```

## Cost Considerations

### Using Claude Code Subscription (Default)

- **Claude Code**: Included with Pro ($20/mo) or Max ($75/mo) subscription
- **Agent SDK with Claude CLI**: **Uses your existing subscription - no additional cost!**
- This is the recommended approach for Pro/Max subscribers

### Using API Key (Optional)

If you choose to use `ANTHROPIC_API_KEY` instead:

- Billed separately based on token usage
  - Input: ~$3 per million tokens
  - Output: ~$15 per million tokens
  - Generating slash commands typically uses modest amounts of tokens
- Monitor your usage at: https://console.anthropic.com/settings/usage
