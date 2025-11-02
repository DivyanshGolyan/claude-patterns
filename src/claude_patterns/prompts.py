"""Shared prompt templates for claude-patterns.

This module contains prompt templates used across different modules,
allowing for consistent prompts and easier filtering of agent-generated messages.
"""

import re


# The full agent prompt template (extracted from generation.py)
# This is used both for generating commands AND for filtering out agent messages
AGENT_GENERATION_PROMPT_TEMPLATE = """I have analyzed user messages and found a cluster of {num_messages} similar messages.
I need you to analyze if they represent a common, reusable pattern that would benefit from a custom slash command for Claude Code.

## How to Write Slash Commands

Slash commands are Markdown files with optional YAML frontmatter. Key information:

**File Format:**
- Filename becomes command name (e.g., `optimize.md` → `/optimize`)
- Content after frontmatter is the prompt Claude executes

**Frontmatter Fields (all optional):**
```yaml
---
description: Brief description of the command (recommended for discoverability)
argument-hint: [message]  # Shows expected arguments during autocomplete
allowed-tools: Bash(git add:*), Bash(git status:*)  # Restrict to specific tools
model: claude-3-5-haiku-20241022  # Use a specific model
---
```

**Argument Placeholders:**
- `$ARGUMENTS` - Captures all passed arguments
- `$1`, `$2`, etc. - Access specific positional arguments

**CRITICAL RULE: If you include `argument-hint`, you MUST use the argument!**
- If you specify `argument-hint`, you MUST include `$ARGUMENTS` or `$1`, `$2`, etc. in your prompt
- The `argument-hint` tells users what to provide, so you need to actually USE what they provide
- Don't add `argument-hint` if you won't reference the argument in your prompt
- This prevents confusing UX where users provide input that gets ignored

**Dynamic Content:**
- `@filename` - Reference files (e.g., "Review @src/utils/helpers.js")
- `` !`command` `` - Execute bash commands inline (e.g., "Current status: !`git status`")

**Good Example (using the argument):**
```markdown
---
description: Create a git commit
argument-hint: [message]
allowed-tools: Bash(git add:*), Bash(git commit:*)
---

Create a git commit with message: $ARGUMENTS

Current status: !`git status`
```

**Bad Example (DON'T do this - has argument-hint but doesn't use it):**
```markdown
---
description: Create a git commit
argument-hint: [message]
---

Create a git commit with an appropriate message based on the changes.
```
☝️ This is WRONG because it asks for [message] but never uses $ARGUMENTS!

## Tools Available

You have access to these tools:
- **Read** - Read existing slash command files to check their content
- **Glob** - List all .md files in the current directory
- **Grep** - Search through files for patterns
- **Write** - Create new slash command files
- **Skip** - Explicitly skip this cluster with a reason (use when NOT creating a command)

## Decision Framework

You have TWO possible outcomes for each cluster:
1. **Create** - Pattern is reusable and no duplicate exists → Use the Write tool to create a slash command
2. **Skip** - Pattern is not reusable OR a duplicate already exists → Use the Skip tool with a clear reason

You MUST use one of these tools. Do not explain your decision in text without using a tool.

## Your Task

Here are the user messages in this cluster:

{sample_messages}

Analyze these messages to determine if they represent a common, reusable pattern.

If they do NOT represent a reusable pattern (too specific, one-off requests, etc.):
- Use the **Skip tool** with a reason explaining why this pattern is not reusable
- Example reason: "Pattern too specific: only applies to this particular project workflow"
- Do NOT create any files or attempt any other action

If they DO represent a reusable pattern:
- FIRST, check for duplicate or similar existing commands:
  1. Use Glob to list all .md files in the current directory
  2. Read existing commands to understand their purpose/description
  3. Compare semantic similarity: Would an existing command serve the same purpose?
  4. If a similar command already exists, use the **Skip tool** with an explanation of which command is similar and why
  5. Only proceed to creation if no duplicate is found
- If no duplicate exists, create a slash command using the Write tool
- Save it to: {output_dir}/[command-name].md
- Use a clear, descriptive kebab-case name (e.g., 'review-code', 'fix-tests')
- Include frontmatter with description
- If users need to provide input, add `argument-hint` AND use `$ARGUMENTS` or `$1`, `$2`, etc. in the prompt
- If no user input is needed, omit `argument-hint` entirely
- Write a clear, actionable prompt that Claude can execute
- Follow the slash command syntax from the documentation
- **CRITICAL: Keep commands CONCISE with HIGH INSTRUCTION DENSITY**
  - Target 3-10 lines of actual instructions (not counting frontmatter)
  - Avoid fluff, filler text, and unnecessary explanations
  - Focus on actionable instructions, not preamble
  - Be direct and specific about what Claude should do
  - Skip pleasantries like "Help me", "I need you to", "Please", etc.
  - Get straight to the point with concrete actions
  - Use bullet points for clarity, not verbose paragraphs
  - Avoid meta-commentary like "What should we do next?" or "If you need more context, ask me"
  - Keep it professional: NO EMOJI, no casual language, no excessive formatting

IMPORTANT ABOUT THE COMMAND CONTENT:
- Write the slash command as instructions TO Claude, not FROM Claude's perspective
- Use imperative/instructional tone: "Check the git configuration..." NOT "I'll check the git configuration..."
- The user will invoke this command, so it should tell Claude what to do
- Think of it as a prompt template that the user is giving to Claude
- When referring to the user's context, use "I" or "my" (from user's perspective) or "you/your" (addressing user)
- NEVER refer to "the user" in third person - the user IS the one speaking

**Example of HIGH instruction density (GOOD):**
```markdown
---
description: Review code for security issues
argument-hint: [file-path]
---

Review $ARGUMENTS for security vulnerabilities:
- SQL injection, XSS, command injection
- Authentication/authorization flaws
- Insecure dependencies
- Sensitive data exposure

Report findings with severity levels and fixes.
```

**Example of LOW instruction density (BAD - too much fluff):**
```markdown
---
description: Review code for security issues
argument-hint: [file-path]
---

I'm working on reviewing code for potential security issues. Help me follow this process:

**Phase 1: Initial Review**
- First, I need you to help me understand what security vulnerabilities might exist
- We should look at things like SQL injection and XSS
- Also consider authentication issues

**Phase 2: Deep Analysis**
- After the initial review, let's dig deeper
- Examine the code more carefully
- Think about edge cases

**Phase 3: Reporting**
- Once we've identified issues, help me document them
- Make sure to explain each vulnerability
- Provide recommendations for how to fix them

What should we look at first? If you need more context, please ask me about the codebase.
```
☝️ This is TOO LONG, too much fluff, low signal. Don't write like this!

Important: Use the Write tool to create the file directly. Do not output the file content as text.
"""


def build_generation_prompt(
    num_messages: int,
    sample_messages: str,
    output_dir: str,
) -> str:
    """Build the agent generation prompt from the template.

    Args:
        num_messages: Number of messages in the cluster
        sample_messages: Formatted string of sample messages
        output_dir: Directory where commands should be saved

    Returns:
        The formatted prompt string
    """
    return AGENT_GENERATION_PROMPT_TEMPLATE.format(
        num_messages=num_messages,
        sample_messages=sample_messages,
        output_dir=output_dir,
    )


def get_prompt_signature() -> str:
    """Get a distinctive signature from the template for filtering.

    Extracts static text chunks from the template (ignoring {placeholders})
    and returns the longest chunk for matching.

    Returns:
        A distinctive string that appears in the agent prompt
    """
    template = AGENT_GENERATION_PROMPT_TEMPLATE

    # Split by {placeholder} patterns to get static text chunks
    static_chunks = re.split(r"\{[^}]+\}", template)

    # Return the longest static chunk (most distinctive)
    return max(static_chunks, key=len).strip()
