"""Shared Rich console instance for ESDC CLI output.

Provides a single Console object that all modules can import
to ensure consistent terminal output (spinner, progress, prints).
"""

from rich.console import Console

console = Console(stderr=True)
