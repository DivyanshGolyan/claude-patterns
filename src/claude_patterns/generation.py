"""Slash command generation using Claude Agent SDK.

This module generates custom slash commands from clustered user messages
by analyzing patterns with Claude AI.
"""

import json
import sys
import os
import subprocess
import anyio
from pathlib import Path
from typing import List, Dict, Any
from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import ToolUseBlock

from claude_patterns.prompts import build_generation_prompt


def check_api_credentials() -> bool:
    """Check if API credentials are available and provide guidance if not.

    Priority:
    1. Check for Claude Code CLI (uses your subscription)
    2. Fall back to ANTHROPIC_API_KEY
    3. Check for CLAUDE_CODE_OAUTH_TOKEN

    Returns:
        True if credentials are found, False otherwise
    """
    # First, check if Claude CLI is available and user is logged in
    try:
        claude_path = subprocess.run(
            ["which", "claude"], capture_output=True, text=True, timeout=5
        )
        if claude_path.returncode == 0 and claude_path.stdout.strip():
            print("Found Claude Code CLI - using your subscription credentials")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Fall back to checking for ANTHROPIC_API_KEY
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and api_key.startswith("sk-ant-"):
        print("Found ANTHROPIC_API_KEY")
        return True

    # Check for CLAUDE_CODE_OAUTH_TOKEN as well
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if oauth_token:
        print("Found CLAUDE_CODE_OAUTH_TOKEN")
        return True

    # No credentials found - provide guidance
    print("Error: No authentication credentials found!\n")
    print("The Claude Agent SDK requires authentication. You have two options:\n")
    print("Option 1: Use Claude Code (recommended for Pro/Max subscribers)")
    print("  - Make sure you're logged in to Claude Code")
    print("  - Run: claude")
    print("  - If not logged in, follow the prompts to authenticate\n")
    print("Option 2: Use an Anthropic API key")
    print("  1. Visit: https://console.anthropic.com/settings/keys")
    print("  2. Create a new API key (starts with 'sk-ant-')")
    print("  3. Set it as an environment variable:")
    print("     export ANTHROPIC_API_KEY='sk-ant-your-key-here'\n")
    print("Note: Option 1 uses your subscription; Option 2 bills separately.")

    return False


def get_commands_dir() -> Path:
    """Get the .claude/commands/ directory in the current working directory.

    Returns:
        Path to .claude/commands/ directory
    """
    cwd = Path.cwd().resolve()
    return cwd / ".claude" / "commands"


async def generate_command_from_cluster(
    cluster_id: int,
    messages: List[Dict[str, Any]],
    output_dir: Path,
    quiet: bool = True,
    max_message_length: int = 500,
    max_messages: int = 20,
) -> bool:
    """Generate a custom slash command for a cluster using Claude Agent SDK.

    Args:
        cluster_id: The cluster identifier
        messages: List of user messages in this cluster
        output_dir: Directory to save the generated slash command
        quiet: If True, suppress verbose output
        max_message_length: Maximum characters per message sent to agent (default: 500)
        max_messages: Maximum number of messages sent to agent per cluster (default: 20)

    Returns:
        True if command was created, False if skipped
    """
    if not quiet:
        print(f"\nProcessing Cluster {cluster_id} ({len(messages)} messages)")

    # Extract just the message text for analysis
    message_texts = [msg.get("message", "") for msg in messages]

    # Build sample messages string (limit messages and truncate each to max length)
    sample_size = min(max_messages, len(message_texts))
    sample_lines = []
    for i, msg in enumerate(message_texts[:sample_size], 1):
        truncated_msg = (
            msg if len(msg) <= max_message_length else msg[:max_message_length] + "..."
        )
        sample_lines.append(f"{i}. {truncated_msg}")

    if len(message_texts) > sample_size:
        sample_lines.append(
            f"\n... and {len(message_texts) - sample_size} more similar messages."
        )

    sample_messages = "\n".join(sample_lines)

    # Build prompt using shared template
    prompt = build_generation_prompt(
        num_messages=len(messages),
        sample_messages=sample_messages,
        output_dir=str(output_dir),
    )

    # Configure options to use Haiku 4.5 with Write tool access
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        allowed_tools=["Write"],
        permission_mode="acceptEdits",
    )

    # Track if Write tool was used
    write_tool_used = False
    created_file_path = None

    # Process the agent's response
    async for message in query(prompt=prompt, options=options):
        try:
            # Check for tool use in assistant messages
            content = getattr(message, "content", None)
            if content is not None:
                for block in content:
                    if isinstance(block, ToolUseBlock) and block.name == "Write":
                        write_tool_used = True
                        created_file_path = block.input.get("file_path")
        except (AttributeError, TypeError):
            # Skip messages without content attribute or non-iterable content
            continue

    # Report the outcome
    if write_tool_used and created_file_path:
        if not quiet:
            command_name = Path(created_file_path).stem
            print(f"  Created: /{command_name}")
        return True
    else:
        if not quiet:
            print("  Skipped")
        return False


async def generate_commands_from_data(
    cluster_metadata: List[Dict[str, Any]],
    cluster_messages: Dict[int, List[Dict[str, Any]]],
    quiet: bool = True,
    max_message_length: int = 500,
    max_messages: int = 20,
) -> int:
    """Generate slash commands from in-memory cluster data.

    Args:
        cluster_metadata: List of cluster metadata dicts (id, size, representative)
        cluster_messages: Dict mapping cluster_id to list of message dicts
        quiet: If True, suppress verbose output
        max_message_length: Maximum characters per message sent to agent
        max_messages: Maximum number of messages sent to agent per cluster

    Returns:
        Number of commands created
    """
    if not cluster_metadata:
        print("Error: No clusters provided", file=sys.stderr)
        return 0

    # Get commands directory and create if needed
    commands_dir = get_commands_dir()
    commands_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    created_count = 0

    # Process each cluster
    for cluster_meta in cluster_metadata:
        cluster_id = cluster_meta["cluster_id"]
        messages = cluster_messages.get(cluster_id, [])

        if not messages:
            continue

        # Generate slash command for this cluster
        was_created = await generate_command_from_cluster(
            cluster_id,
            messages,
            commands_dir,
            quiet=quiet,
            max_message_length=max_message_length,
            max_messages=max_messages,
        )
        if was_created:
            created_count += 1

    return created_count


async def generate_commands(
    clusters_dir: Path,
    quiet: bool = True,
    max_message_length: int = 500,
    max_messages: int = 20,
) -> int:
    """Generate slash commands from all cluster files.

    Args:
        clusters_dir: Directory containing cluster JSON files
        quiet: If True, suppress verbose output
        max_message_length: Maximum characters per message sent to agent (default: 500)
        max_messages: Maximum number of messages sent to agent per cluster (default: 20)

    Returns:
        Number of commands created
    """
    # Find all cluster JSON files
    cluster_files = sorted(clusters_dir.glob("cluster_*.json"))

    if not cluster_files:
        print(f"Error: No cluster files found in {clusters_dir}", file=sys.stderr)
        return 0

    # Get commands directory and create if needed
    commands_dir = get_commands_dir()
    commands_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    created_count = 0

    # Process each cluster
    for cluster_file in cluster_files:
        # Extract cluster ID from filename
        cluster_id = int(cluster_file.stem.split("_")[1])

        # Load cluster messages
        with open(cluster_file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        # Guard: Skip empty clusters
        if not messages:
            continue

        # Generate slash command for this cluster
        was_created = await generate_command_from_cluster(
            cluster_id,
            messages,
            commands_dir,
            quiet=quiet,
            max_message_length=max_message_length,
            max_messages=max_messages,
        )
        if was_created:
            created_count += 1

    return created_count


async def main():
    """CLI entry point for slash command generation."""
    # Check for API credentials first
    if not check_api_credentials():
        sys.exit(1)

    if len(sys.argv) < 2:
        print(
            "Usage: generate-claude-commands <clusters_folder>",
            file=sys.stderr,
        )
        sys.exit(1)

    clusters_dir = Path(sys.argv[1])

    if not clusters_dir.exists():
        print(f"Error: Directory '{clusters_dir}' not found", file=sys.stderr)
        sys.exit(1)

    if not clusters_dir.is_dir():
        print(f"Error: '{clusters_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Generate commands
    created_count = await generate_commands(clusters_dir, quiet=False)

    if created_count > 0:
        print(f"\nGenerated {created_count} slash command(s)")
        print(f"Location: {get_commands_dir()}")
    else:
        print("\nNo reusable patterns found.")


def cli_main():
    """Wrapper for anyio to run async main."""
    anyio.run(main)
