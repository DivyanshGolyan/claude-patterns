"""Output utilities: timing, progress indicators, verbosity control."""

import sys
import time
from enum import Enum
from contextlib import contextmanager


class OutputLevel(Enum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


_output_level = OutputLevel.NORMAL


def set_output_level(level: OutputLevel) -> None:
    global _output_level
    _output_level = level


def get_output_level() -> OutputLevel:
    return _output_level


def should_print(min_level: OutputLevel = OutputLevel.NORMAL) -> bool:
    return _output_level.value >= min_level.value


@contextmanager
def timed_phase(description: str, min_level: OutputLevel = OutputLevel.NORMAL):
    if should_print(min_level):
        print(f"→ {description}...", flush=True)

    start = time.time()
    yield
    elapsed = time.time() - start

    if should_print(min_level):
        print(f"  Completed in {elapsed:.1f}s")


def print_phase(message: str, min_level: OutputLevel = OutputLevel.NORMAL) -> None:
    if should_print(min_level):
        print(message, flush=True)


def print_info(message: str, min_level: OutputLevel = OutputLevel.NORMAL) -> None:
    if should_print(min_level):
        print(message)


def print_verbose(message: str) -> None:
    if should_print(OutputLevel.VERBOSE):
        print(message)


def print_error(message: str) -> None:
    print(message, file=sys.stderr)


class ProgressCounter:
    """Simple progress counter displaying 'X of Y' progress."""

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        min_level: OutputLevel = OutputLevel.NORMAL,
    ):
        self.total = total
        self.description = description
        self.current = 0
        self.min_level = min_level
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        self.current += increment
        if should_print(self.min_level):
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            print(
                f"  {self.description} {self.current}/{self.total}... "
                f"({rate:.1f} items/s)",
                end="\r",
                flush=True,
            )

    def finish(self) -> None:
        if should_print(self.min_level):
            elapsed = time.time() - self.start_time
            print(
                f"  {self.description} {self.total}/{self.total} "
                f"- completed in {elapsed:.1f}s"
            )


def format_time(seconds: float) -> str:
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
