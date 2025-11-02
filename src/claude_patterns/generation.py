"""Slash command generation using Claude Agent SDK.

This module generates custom slash commands from clustered user messages
by analyzing patterns with Claude AI.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    tool,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import ToolUseBlock

from claude_patterns.prompts import build_generation_prompt
from claude_patterns.output import (
    print_info,
    print_verbose,
    ProgressCounter,
)


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


def create_control_tools_server():
    """Create a custom MCP server with control flow tools.

    Returns:
        An MCP server with the 'skip' tool for the agent to explicitly skip clusters.
    """

    @tool("skip", "Skip creating a slash command for this cluster", {"reason": str})
    async def skip_command(args: Dict[str, Any]) -> Dict[str, Any]:
        """Skip creating a command with an explanation.

        Args:
            args: Dictionary with 'reason' key containing skip explanation

        Returns:
            Tool response confirming skip
        """
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Cluster skipped: {args.get('reason', 'No reason provided')}",
                }
            ]
        }

    return create_sdk_mcp_server(name="control", tools=[skip_command])


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
    max_message_length: int = 500,
    max_messages: int = 20,
) -> tuple[bool, str | None]:
    """Generate a custom slash command for a cluster using Claude Agent SDK.

    Args:
        cluster_id: The cluster identifier
        messages: List of user messages in this cluster
        output_dir: Directory to save the generated slash command
        max_message_length: Maximum characters per message sent to agent (default: 500)
        max_messages: Maximum number of messages sent to agent per cluster (default: 20)

    Returns:
        Tuple of (was_created, command_name_or_none)
    """
    print_verbose(f"\nProcessing Cluster {cluster_id} ({len(messages)} messages)")

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

    # Configure options to use Haiku 4.5 with read/write access to commands directory
    # Set cwd to commands_dir so agent operates in that context for duplicate detection
    # Include custom skip tool for explicit skip decisions
    control_server = create_control_tools_server()
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        cwd=str(output_dir),
        mcp_servers={"control": control_server},
        allowed_tools=["Read", "Glob", "Grep", "Write", "mcp__control__skip"],
        permission_mode="acceptEdits",
    )

    # Track Write tool usage for command creation and skip tool calls
    write_tool_used = False
    created_file_path = None
    skip_tool_used = False
    skip_reason = None

    # Use ClaudeSDKClient to support custom tools
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            try:
                # Check for tool use in assistant messages
                content = getattr(message, "content", None)
                if content is not None:
                    for block in content:
                        # Track Write tool usage for command creation
                        if isinstance(block, ToolUseBlock):
                            if block.name == "Write":
                                write_tool_used = True
                                created_file_path = block.input.get("file_path")
                            # Check for skip tool invocation (could be "skip" or "mcp__control__skip")
                            elif block.name in ("skip", "mcp__control__skip"):
                                skip_tool_used = True
                                skip_reason = block.input.get(
                                    "reason", "No reason provided"
                                )
            except (AttributeError, TypeError):
                # Skip messages without content attribute or non-iterable content
                continue

    # Report the outcome
    if write_tool_used and created_file_path:
        command_name = Path(created_file_path).stem
        print_info(f"    ✓ Created slash command: /{command_name}")
        return True, command_name
    elif skip_tool_used:
        # Agent explicitly used skip tool with a reason
        print_info(f"    ⊘ Skipped: {skip_reason}")
        return False, None
    else:
        # Agent didn't use either tool - this shouldn't happen with updated prompt
        print_info("    ✗ Skipped (agent did not make explicit decision)")
        return False, None


async def generate_commands_from_data(
    cluster_metadata: List[Dict[str, Any]],
    cluster_messages: Dict[int, List[Dict[str, Any]]],
    max_message_length: int = 500,
    max_messages: int = 20,
) -> tuple[int, List[str]]:
    """Generate slash commands from in-memory cluster data.

    Args:
        cluster_metadata: List of cluster metadata dicts (id, size, representative)
        cluster_messages: Dict mapping cluster_id to list of message dicts
        max_message_length: Maximum characters per message sent to agent
        max_messages: Maximum number of messages sent to agent per cluster

    Returns:
        Tuple of (number of commands created, list of command names created)
    """
    if not cluster_metadata:
        print("Error: No clusters provided", file=sys.stderr)
        return 0, []

    # Get commands directory and create if needed
    commands_dir = get_commands_dir()
    commands_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    created_count = 0
    created_commands = []
    skipped_count = 0
    total_clusters = len(cluster_metadata)

    # Create progress counter
    progress = ProgressCounter(total_clusters, "Processing clusters")

    # Process each cluster
    for cluster_meta in cluster_metadata:
        cluster_id = cluster_meta["cluster_id"]
        messages = cluster_messages.get(cluster_id, [])

        if not messages:
            progress.update()
            continue

        # Generate slash command for this cluster
        was_created, command_name = await generate_command_from_cluster(
            cluster_id,
            messages,
            commands_dir,
            max_message_length=max_message_length,
            max_messages=max_messages,
        )
        if was_created and command_name:
            created_count += 1
            created_commands.append(command_name)
        else:
            skipped_count += 1

        progress.update()

    progress.finish()

    # Print summary of what was generated vs skipped
    summary_parts = []
    if created_count > 0:
        summary_parts.append(f"✓ Created {created_count}")
    if skipped_count > 0:
        summary_parts.append(f"✗ Skipped {skipped_count}")

    if summary_parts:
        print_verbose(f"  {', '.join(summary_parts)} (see details above)")
    else:
        print_verbose("  No commands generated")

    return created_count, created_commands
