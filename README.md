# ESDC Data Management Module

This module provides functionality for managing data related to the ESDC (https://esdc.skkmigas.go.id). It includes commands for fetching, validating, and displaying data from various resources, as well as loading data into a SQLite database. The module utilizes the Typer library for command-line interface (CLI) interactions and Rich for enhanced logging and output formatting.

## Key Features

- **Fetch Data**: Fetch data from the ESDC API in various formats (CSV, JSON, ZIP).
- **Load Data**: Load data into a SQLite database.
- **Validate Data**: Validate data against predefined rules.
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
pip install requests==2.32.2 python-dotenv==1.0.1 pandas==2.2.2 tqdm==4.66.4 typer==0.12.3 platformdirs==4.2.2 tabulate==0.9.0 ollama==0.3.0 mypy==1.11.0 openpyxl==3.1.5
```
## Usage

Run the module from the command line to access the available commands and options. Below are the main commands provided by the module:

### `init`
Initializes the application and fetches data.

### `fetch`

Downloads data from the ESDC API and saves it to a specified file type.

```sh
esdc fetch --filetype csv --save
```

Options:

- `filetype`: The type of file to save the data to. Options are "csv" or "json". Defaults to "json".
- `save`: Indicates whether to save the data to a file. Defaults to `False`.

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
columns: (Optional) A space-separated list of columns to select. Defaults to "".

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

For any questions or issues, please contact me at fajril at ambia.id.