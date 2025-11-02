"""Main CLI orchestration for claude-patterns pipeline.

This module provides the main entry point that orchestrates the complete pipeline:
1. Extract user messages from Claude Code conversation files
2. Cluster similar messages using semantic similarity
3. Generate slash commands from the clusters
"""

import sys
import argparse
import anyio
from pathlib import Path
from typing import Optional

# Import our modules
from claude_patterns.extraction import extract_all_messages
from claude_patterns.clustering import (
    compute_embeddings,
    cluster_messages,
    prepare_clusters,
)
from claude_patterns.generation import (
    check_api_credentials,
    generate_commands_from_data,
)


def find_conversations_folder() -> Optional[Path]:
    """Auto-detect Claude Code conversations folder for current project.

    Claude Code stores conversations in:
    ~/.claude/projects/<encoded-path>/[session-uuid].jsonl

    Where <encoded-path> is the absolute project path with / replaced by -
    Example: /Users/name/Documents/GitHub/project -> -Users-name-Documents-GitHub-project

    Returns:
        Path to conversations folder if found, None otherwise
    """
    cwd = Path.cwd().resolve()

    # Claude Code encodes project paths by replacing / with -
    home_claude = Path.home() / ".claude" / "projects"
    if not home_claude.exists():
        return None

    # Encode current path the same way Claude Code does
    encoded_path = str(cwd).replace("/", "-")
    project_folder = home_claude / encoded_path

    # Check if this project folder exists and contains .jsonl conversation files
    if project_folder.exists() and project_folder.is_dir():
        # Verify it contains at least one .jsonl file
        if any(project_folder.glob("*.jsonl")):
            return project_folder

    return None


async def run_pipeline(
    conversations_folder: Path,
    min_cluster_size: int = 2,
    threshold: float = 0.7,
    model: str = "all-MiniLM-L6-v2",
    max_message_length: int = 500,
    max_messages: int = 20,
) -> None:
    """Run the complete pipeline to generate slash commands.

    Args:
        conversations_folder: Directory containing JSONL conversation files
        min_cluster_size: Minimum messages per cluster (default: 2)
        threshold: Distance threshold for clustering (default: 0.7)
        model: Sentence-transformer model to use (default: all-MiniLM-L6-v2)
        max_message_length: Maximum characters per message sent to agent (default: 500)
        max_messages: Maximum number of messages sent to agent per cluster (default: 20)
    """
    # Step 1: Extract user messages (in-memory)
    print("→ Extracting user messages...")
    all_messages = extract_all_messages(
        conversations_folder, exclude_system=True, verbose=False
    )

    if not all_messages:
        print("Error: No user messages found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(all_messages)} user messages")

    # Step 2: Cluster similar messages (in-memory)
    print("→ Clustering similar messages...")
    embedding_model, embeddings = compute_embeddings(all_messages, model)
    clusters, labels = cluster_messages(all_messages, embeddings, threshold)
    cluster_metadata, clusters_data = prepare_clusters(
        clusters,
        all_messages,
        embeddings,
        labels,
        embedding_model,
        min_size=min_cluster_size,
    )

    if not cluster_metadata:
        print("\nNo significant clusters found.")
        return

    # Step 3: Generate slash commands (only writes final .md files)
    print("→ Generating slash commands...")
    created_count = await generate_commands_from_data(
        cluster_metadata,
        clusters_data,
        quiet=True,
        max_message_length=max_message_length,
        max_messages=max_messages,
    )

    # Final summary
    commands_dir = Path.cwd() / ".claude" / "commands"
    command_files = list(commands_dir.glob("*.md")) if commands_dir.exists() else []

    print("\nDone!\n")

    if created_count > 0 and command_files:
        print(f"Generated {created_count} slash command(s):")
        for cmd_file in sorted(command_files):
            cmd_name = cmd_file.stem
            print(f"  • /{cmd_name}")
        print(f"\nLocation: {commands_dir}")
        print("Reload Claude Code to use the new commands\n")
    else:
        print("No reusable patterns found in your conversations.\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate custom slash commands from Claude Code conversation history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                               # Auto-detect conversations folder
  %(prog)s ~/conversations               # Specify conversations folder
  %(prog)s ~/conversations --min-cluster-size 3

Auto-detection:
  Searches for ~/.claude/projects/<encoded-path>/
  where <encoded-path> is the current directory with / replaced by -
  Example: /Users/name/project → -Users-name-project

This script will:
  1. Extract user messages from JSONL files
  2. Cluster similar messages
  3. Generate slash commands for common patterns
        """,
    )

    parser.add_argument(
        "conversations_folder",
        type=Path,
        nargs="?",
        help="Directory containing JSONL conversation files (auto-detected if omitted)",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum number of messages per cluster (default: 2)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Distance threshold for clustering (default: 0.7, lower = stricter)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--max-message-length",
        type=int,
        default=500,
        help="Maximum characters per message sent to agent (default: 500)",
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Maximum number of messages sent to agent per cluster (default: 20)",
    )

    args = parser.parse_args()

    # Auto-detect conversations folder if not provided
    if args.conversations_folder is None:
        detected_folder = find_conversations_folder()
        if detected_folder is None:
            cwd = Path.cwd().resolve()
            encoded_path = str(cwd).replace("/", "-")
            expected_path = Path.home() / ".claude" / "projects" / encoded_path

            print(
                "Error: Could not auto-detect Claude Code conversations",
                file=sys.stderr,
            )
            print(f"\nCurrent directory: {cwd}", file=sys.stderr)
            print(f"Expected location: {expected_path}", file=sys.stderr)

            if not expected_path.exists():
                print(
                    "\nThe Claude Code project folder doesn't exist.", file=sys.stderr
                )
                print(
                    "   This usually means Claude Code hasn't been used in this directory yet.",
                    file=sys.stderr,
                )
            elif not any(expected_path.glob("*.jsonl")):
                print(
                    "\nNo conversation files (.jsonl) found in the project folder.",
                    file=sys.stderr,
                )
                print(
                    "   Try running Claude Code in this directory first to create conversations.",
                    file=sys.stderr,
                )

            print(
                "\nAlternatively, specify the conversations folder explicitly:",
                file=sys.stderr,
            )
            print(
                "  claude-patterns ~/.claude/projects/-path-to-your-project",
                file=sys.stderr,
            )
            sys.exit(1)
        args.conversations_folder = detected_folder

    # Validate conversations folder
    if not args.conversations_folder.exists():
        print(
            f"Error: Directory '{args.conversations_folder}' not found", file=sys.stderr
        )
        sys.exit(1)

    if not args.conversations_folder.is_dir():
        print(
            f"Error: '{args.conversations_folder}' is not a directory", file=sys.stderr
        )
        sys.exit(1)

    # Check API credentials before starting
    if not check_api_credentials():
        sys.exit(1)

    # Display header
    print("\nClaude Code Slash Command Generator")
    print(f"   Project: {Path.cwd().name}")
    print(
        f"   Conversations: {len(list(args.conversations_folder.glob('*.jsonl')))} files\n"
    )

    # Run the pipeline
    anyio.run(
        run_pipeline,
        args.conversations_folder,
        args.min_cluster_size,
        args.threshold,
        args.model,
        args.max_message_length,
        args.max_messages,
    )
