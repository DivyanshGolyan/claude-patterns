"""Output utilities for CLI feedback and timing.

This module provides utilities for:
- Timing operations and displaying elapsed time
- Progress indicators
- Verbosity level management
"""

import sys
import time
from enum import Enum
from contextlib import contextmanager


class OutputLevel(Enum):
    """Output verbosity levels."""

    QUIET = 0  # Only final results
    NORMAL = 1  # Default: essential information
    VERBOSE = 2  # All details including debugging info


# Global output level (can be set by CLI)
_output_level = OutputLevel.NORMAL


def set_output_level(level: OutputLevel) -> None:
    """Set the global output verbosity level."""
    global _output_level
    _output_level = level


def get_output_level() -> OutputLevel:
    """Get the current output verbosity level."""
    return _output_level


def should_print(min_level: OutputLevel = OutputLevel.NORMAL) -> bool:
    """Check if output should be printed at the given level.

    Args:
        min_level: Minimum level required to print

    Returns:
        True if current level >= min_level
    """
    return _output_level.value >= min_level.value


@contextmanager
def timed_phase(description: str, min_level: OutputLevel = OutputLevel.NORMAL):
    """Context manager for timing a phase and displaying elapsed time.

    Args:
        description: Description of the phase (e.g., "Extracting user messages")
        min_level: Minimum output level required to display this phase

    Example:
        with timed_phase("Extracting user messages"):
            # do work
            pass
        # Output: → Extracting user messages... (0.3s)
    """
    if should_print(min_level):
        print(f"→ {description}...", flush=True)

    start = time.time()
    yield
    elapsed = time.time() - start

    if should_print(min_level):
        # Print elapsed time on the same line after the operation
        print(f"  Completed in {elapsed:.1f}s")


def print_phase(message: str, min_level: OutputLevel = OutputLevel.NORMAL) -> None:
    """Print a phase indicator message.

    Args:
        message: Message to print
        min_level: Minimum output level required to display
    """
    if should_print(min_level):
        print(message, flush=True)


def print_info(message: str, min_level: OutputLevel = OutputLevel.NORMAL) -> None:
    """Print an informational message.

    Args:
        message: Message to print
        min_level: Minimum output level required to display
    """
    if should_print(min_level):
        print(message)


def print_verbose(message: str) -> None:
    """Print a message only in verbose mode.

    Args:
        message: Message to print
    """
    if should_print(OutputLevel.VERBOSE):
        print(message)


def print_error(message: str) -> None:
    """Print an error message (always shown, even in quiet mode).

    Args:
        message: Error message to print
    """
    print(message, file=sys.stderr)


class ProgressCounter:
    """Simple progress counter for displaying 'X of Y' progress.

    Example:
        progress = ProgressCounter(40, "Processing cluster")
        for item in items:
            progress.update()
            # do work
        # Output: Processing cluster 15/40...
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        min_level: OutputLevel = OutputLevel.NORMAL,
    ):
        """Initialize progress counter.

        Args:
            total: Total number of items
            description: Description of the operation
            min_level: Minimum output level to display progress
        """
        self.total = total
        self.description = description
        self.current = 0
        self.min_level = min_level
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        """Update progress counter.

        Args:
            increment: Number of items to increment by (default: 1)
        """
        self.current += increment
        if should_print(self.min_level):
            # Print progress on the same line using carriage return
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            print(
                f"  {self.description} {self.current}/{self.total}... "
                f"({rate:.1f} items/s)",
                end="\r",
                flush=True,
            )

    def finish(self) -> None:
        """Mark progress as complete and move to next line."""
        if should_print(self.min_level):
            elapsed = time.time() - self.start_time
            print(
                f"  {self.description} {self.total}/{self.total} "
                f"- completed in {elapsed:.1f}s"
            )


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1.5s", "2m 15s", "1h 5m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
