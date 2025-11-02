#!/usr/bin/env python3
"""
Test to see what the SDK returns.
"""

import anyio
from claude_agent_sdk import query, ClaudeAgentOptions


async def test_sdk():
    """Test SDK response format."""
    print("Testing SDK response format...\n")

    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        max_turns=1,
    )

    async for message in query(
        prompt="Write a short haiku about coding.", options=options
    ):
        print(f"Message type: {type(message)}")
        print(f"Message content: {message}")
        print("---")


if __name__ == "__main__":
    anyio.run(test_sdk)
