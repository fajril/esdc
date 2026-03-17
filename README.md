# ESDC Data Management Module

This module provides functionality for managing data related to the ESDC (https://esdc.skkmigas.go.id). It includes commands for fetching and displaying data from various resources, as well as loading data into a SQLite database. The module utilizes the Typer library for command-line interface (CLI) interactions and Rich for enhanced logging and output formatting.

## Key Features

- **Fetch Data**: Fetch data from the ESDC API in various formats (CSV, JSON, ZIP).
- **Load Data**: Load data into a SQLite database.
- **Display Data**: Display data from specific tables with filtering options.
- **Save Data**: Save output data to files.

## Installation

To install the module, clone the repository and install the dependencies specified in the `pyproject.toml` file.

```sh
git clone https://github.com/fajril/esdc.git
cd esdc
pip install .
```

Alternatively, you can install the dependencies directly:

```sh
pip install requests==2.32.2 pandas==2.2.2 tqdm==4.66.4 typer>=0.12.4 tabulate==0.9.0 ollama==0.3.0 basedpyright>=0.1.0 openpyxl==3.1.5 pyyaml>=6.0
```

## Configuration

On first run, the application creates a configuration directory at `~/.esdc/` with a default `config.yaml` file.

### Config File Location

```
~/.esdc/
├── config.yaml    # Configuration settings
└── esdc.db       # Database file
```

### Config File Structure

```yaml
api_url: https://esdc.skkmigas.go.id/
database_path: ~/.esdc/esdc.db
```

### Interactive Credentials

The first time you run a command that requires authentication, you will be prompted to enter your username and password. These credentials are not stored - you will be prompted again on subsequent runs.

For automated workflows (CI/CD), you can set environment variables:

```bash
export ESDC_USER="your_username"
export ESDC_PASS="your_password"
```

### Environment Variables

| Variable | Purpose | Priority |
|----------|---------|----------|
| `ESDC_USER` | API username | Env > Interactive prompt |
| `ESDC_PASS` | API password | Env > Interactive prompt |
| `ESDC_URL` | API URL | Env > config.yaml > Default |
| `ESDC_DB_FILE` | Database file path | Env > config.yaml > Default |
| `ESDC_CONFIG_DIR` | Config directory | Env > Default `~/.esdc` |

### Commands

```bash
# Show database location
esdc db-info
```

## Usage

Run the module from the command line to access the available commands and options. Below are the main commands provided by the module:

### `fetch`

Downloads data from the ESDC API and saves it to a specified file type.

```sh
esdc fetch --filetype csv --save
```

Options:

- `filetype`: The type of file to save the data to. Options are "csv" or "json". Defaults to "json".
- `save`: Indicates whether to save the fetched data to a file. Defaults to `False`.

### `reload`

Reloads data from existing files into the database.

```sh
esdc reload --filetype csv
```

Options:

- `filetype`: The type of file to save the data to. Options are "csv", "json", or "zip". Defaults to "json".

### `show`

Displays data from a specified table with optional filters.

```sh
esdc show table_name --where column_name --search filter_value --year 2023 --output 0 --save --columns column1 column2
```

Arguments:

- table: The name of the table to show data from.
- where: (Optional) The column to search. Defaults to None.
- search: (Optional) A search keyword to apply to the selected column in where clause. Defaults to "".
- year: (Optional) A filter year value to apply to the data. Defaults to None.
- output: (Optional) The level of detail to show in the output. Defaults to 0.
- save: (Optional) Whether to save the output data to a file. Defaults to False.
- columns: (Optional) A space-separated list of columns to select. Defaults to "".

### Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your branch to your fork.
5. Create a pull request on the main repository.

### License
This project is licensed under the terms of the Apache Software license. See the `LICENSE` file for details.

### Contact

For any questions or issues, please contact me at fambia at skkmigas.go.id or fajril at ambia.id.
