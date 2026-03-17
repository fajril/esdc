# Contributing to ESDC

Thank you for your interest in contributing to ESDC!

## Project Overview

ESDC is a Python CLI package for managing data from the ESDC API (https://esdc.skkmigas.go.id). It provides commands for fetching, validating, and displaying data from various resources, with SQLite database storage.

## Build/Lint/Test Commands

### Installation
```bash
pip install -e .
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/test_idgen.py

# Run single test function
pytest tests/test_idgen.py::test_luhn_mod16_known_payloads

# Run with verbose output
pytest -v tests/
```

### Linting and Type Checking
```bash
# Run pylint on source
pylint esdc/

# Run basedpyright type checking
basedpyright esdc/
```

### Running the CLI
```bash
# Show help
esdc --help

# Run with verbose logging (0=warning, 1=info, 2+=debug)
esdc -vv show project_resources
esdc fetch --filetype csv --save
```

## Code Style Guidelines

### Imports

Organize imports in three groups, separated by blank lines:

1. Standard library imports (alphabetically)
2. Third-party imports (alphabetically)
3. Local imports (from esdc...)

```python
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from rich import print
import typer

from esdc.configs import Config
from esdc.selection import TableName
```

### Type Hints

Use Python 3.10+ style type hints:

- Use `list[str]` instead of `List[str]`
- Use `dict[str, int]` instead of `Dict[str, int]`
- Use `X | None` instead of `Optional[X]`
- Always annotate function parameters and return types

```python
def process_data(items: list[str], count: int = 0) -> dict[str, int] | None:
    """Process data items."""
    pass
```

### Docstrings

Use Google/NumPy style docstrings with these sections:

- One-line summary
- Extended description (if needed)
- Args/Parameters
- Returns
- Raises (if applicable)
- Notes (if applicable)

```python
def load_data(file_path: str, validate: bool = True) -> pd.DataFrame:
    """Load data from a file and optionally validate it.

    Args:
        file_path: Path to the data file.
        validate: Whether to validate the loaded data. Defaults to True.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validation fails.
    """
```

### Naming Conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Private functions: Prefix with underscore (`_private_function`)
- Enum classes: `PascalCase` for class, `UPPER_CASE` for members

### Error Handling

- Use specific exception types, not bare `except`
- Log errors using the logging module
- Return `None` for optional results rather than raising when appropriate

```python
try:
    result = perform_operation()
except ValueError as e:
    logging.error("Operation failed: %s", str(e))
    return None
```

### CLI Commands

Use Typer decorators for CLI commands:

```python
from typing_extensions import Annotated

@app.command()
def show(
    table: Annotated[str, typer.Argument(help="Table name.")],
    where: Annotated[str | None, typer.Option(help="Column to search.")] = None,
) -> None:
    """Brief description of command."""
    ...
```

### Dataclasses for Configuration

Use frozen dataclasses for immutable configuration:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    APP_NAME: str = "esdc"
    VERSION: str = "0.2.1"
```

### Enums for Constants

Use Enum classes for grouped constants:

```python
from enum import Enum

class TableName(Enum):
    PROJECT_RESOURCES = "project_resources"
    FIELD_RESOURCES = "field_resources"
```

## Project Structure

```
esdc/
├── __init__.py       # Package exports
├── esdc.py           # Main CLI application
├── configs.py        # Configuration dataclass
├── selection.py      # Enum classes for constants
├── idgen.py          # ID generation utilities
├── dbmanager.py      # Database operations
└── sql/              # SQL query files

tests/
├── test_idgen.py     # Tests for idgen module
└── ...
```

## Key Dependencies

- **typer**: CLI framework
- **pandas**: Data manipulation
- **rich**: Terminal output formatting
- **requests**: HTTP client
- **pytest**: Testing framework

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes following the code style guidelines
4. Run tests and type checking
5. Commit with clear commit messages
6. Push to your fork
7. Open a pull request

## Questions?

For questions or discussions, please open an issue on GitHub.
