#!/usr/bin/env python3
"""
Quick test to see if Claude Agent SDK can use local Claude Code credentials.
"""

import anyio
from claude_agent_sdk import query


async def test_sdk():
    """Test if SDK works with existing Claude Code credentials."""
    print("Testing Claude Agent SDK with local credentials...")
    print("Sending a simple query to Claude...\n")

    try:
        response_count = 0
        async for message in query(
            prompt="What is 2 + 2? Just answer with the number."
        ):
            response_count += 1
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    print(content, end="")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block.get("text", ""), end="")

        print("\n\nSuccess! The SDK is working with your Claude Code credentials.")
        print("No separate API key needed!")
        return True

    except Exception as e:
        print(f"\nError: {e}")
        print("\nThe SDK could not connect. You may need to set ANTHROPIC_API_KEY.")
        return False


if __name__ == "__main__":
    anyio.run(test_sdk)
