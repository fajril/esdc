"""
ESDC Data Management Module

This module provides functionality for managing data 
related to the ESDC (https://esdc.skkmigas.go.id). 
It includes commands for fetching, validating, and displaying data from various resources,
as well as loading data into a SQLite database. 
The module utilizes the Typer library for command-line interface (CLI) interactions 
and Rich for enhanced logging and output formatting.

Key Features:
- Fetch data from the ESDC API in various formats (CSV, JSON, ZIP).
- Load data into a SQLite database.
- Validate data against predefined rules.
- Display data from specific tables with filtering options.
- Save output data to files.

Dependencies:
- pandas: For data manipulation and storage.
- requests: For making HTTP requests to the ESDC API.
- rich: For enhanced terminal output and logging.
- dotenv: For loading environment variables from a .env file.
- sqlite3: For database operations.

Commands:
- init: Initializes the application and fetches data.
- fetch: Downloads data from the ESDC API and saves it to a specified file type.
- reload: Reloads data from existing binary files into the database.
- show: Displays data from a specified table with optional filters.
- validate: Validates data from a file or performs a full validation.

Usage:
Run the module from the command line to access the available commands and options.
"""

import os
import io
from datetime import date
from pathlib import Path
import csv
import json
import gzip
from contextlib import closing
import logging
from typing import Union, List, Tuple, Iterable, Optional

import requests
from dotenv import load_dotenv, find_dotenv
import typer
from typing_extensions import Annotated
import rich
from rich.progress import Progress, TransferSpeedColumn, DownloadColumn
from rich.logging import RichHandler
from rich.prompt import Prompt
import pandas as pd
from tabulate import tabulate

from esdc.selection import TableName, ApiVer, FileType
from esdc.validate import RuleEngine
from esdc.configs import Config
from esdc.summarizer import describer
from esdc.dbmanager import run_query, load_data_to_db

TABLES: Tuple[TableName, TableName] = (
    TableName.PROJECT_RESOURCES,
    TableName.PROJECT_TIMESERIES,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main(verbose: int = 0):
    """
    Main function to set up logging and log level.

    Args:
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        None

    Notes:
        This function sets up the logging configuration using the RichHandler.
        The log level is set based on the verbosity level:
            - verbose >= 2: DEBUG
            - verbose == 1: INFO
            - verbose == 0: WARNING
    """
    load_dotenv(find_dotenv())
    handler = RichHandler(show_time=False)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logger = logging.getLogger()
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.info("Log level set to %s", logging.getLevelName(logger.getEffectiveLevel()))


@app.command()
def fetch(
    filetype: str = typer.Option("csv", help="Options: csv, json"),
    save: bool = typer.Option(
        False,
        "--save/--no-save",
        help="Specify whether to save the fetched data to a file.",
    ),
) -> None:
    """
    Fetch data from ESDC and save it to a file.

    This function fetches data from the ESDC API and saves it to a file
    based on the specified file type. If the file type is not supported,
    a warning will be logged.

    Parameters:
        filetype:
            The type of file to save the data to. Options are "csv" or "json".
            Defaults to "csv".
        save:
            Indicates whether to save the data to a file. Defaults to False.

    Returns:
        None
    """

    env_available = load_dotenv(find_dotenv())
    if env_available:
        username = os.getenv("ESDC_USER") or ""
        password = os.getenv("ESDC_PASS") or ""
    else:
        logging.debug("Environment variables is not found.")
        username = ""
        password = ""

    if not username:
        logging.debug("Requesting credential from user.")
        username = Prompt.ask("user")
        password = Prompt.ask("pass", password=True)

    if filetype == "csv":
        load_esdc_data(
            filetype=FileType.CSV, to_file=save, username=username, password=password
        )
    elif filetype == "json":
        load_esdc_data(
            filetype=FileType.JSON, to_file=save, username=username, password=password
        )
    else:
        logging.warning("File type %s is not available.", filetype)


@app.command()
def reload(
    filetype: Annotated[
        Optional[str], typer.Option(help="Options: csv, json, zip")
    ] = "csv"
) -> None:
    """
    Reload data from binary files and save it to a file.

    Args:
        filetype (str, optional): The type of file to save the data to. Defaults to "csv".

    Returns:
        None

    Notes:
        This function reloads data from binary files and saves it
        to a file based on the provided filetype.
        If the filetype is not supported, a debug message will be printed.
        If a file is not found, a warning message will be printed.
    """
    for table in TABLES:
        if Path(f"{table.value}.{filetype}").exists():
            filename = f"{table.value}.{filetype}"
            if filetype == "csv":
                _load_file_as_csv(filename, table.value)
            elif filetype == "json":
                _load_file_as_json(filename, table.value)
            else:
                logging.debug(
                    "failed to load %s. Unknown %s format", filename, filetype
                )
        else:
            logging.warning("File %s is not found.", filename)


@app.command()
def show(
    table: Annotated[str, typer.Argument(help="Table name.")],
    where: Annotated[Optional[str], typer.Option(help="Column to search.")] = None,
    search: Annotated[Optional[str], typer.Option(help="Filter value")] = "",
    year: Annotated[
        Optional[int], typer.Option(min=2019, help="Filter year value")
    ] = None,
    output: Annotated[int, typer.Option(help="Detail of output.")] = 0,
    save: bool = typer.Option(
        False,
        "--save/--no-save",
        help="Specify whether to save the shown data to a file.",
    ),
    columns: Annotated[str, typer.Option(help="select column")] = "",
):
    """
    Show data from a specific table.

    Args:
        table (str): The name of the table to show data from.
        where (str, optional): The column to search. Defaults to None.
        search (str, optional): A search keyword to apply to the selected column in where clause. 
            Defaults to "".
        year (int, optional): A filter year value to apply to the data. Defaults to None.
        output (int, optional): The level of detail to show in the output. Defaults to 0.
        save (bool, optional): Whether to save the output data to a file. Defaults to False.
        columns (str, optional): A space-separated list of column(s) to select. Defaults to "".

    Returns:
        None

    Notes:
        This function runs a query on the specified table with the provided filters
        and displays the result.
        The output is formatted to display float values with two decimal places.
        If the save option is True, the output data will be saved to a CSV file.
    """
    if columns.strip():
        columns_splitted: Union[List[str], str] = columns.split(" ")
    else:
        columns_splitted = ""
    df = run_query(
        table=TableName(table),
        where=where,
        like=search,
        year=year,
        output=output,
        columns=columns_splitted,
    )
    if df is not None:
        pd.options.display.float_format = "{:,.2f}".format
        formatted_table = tabulate(
            df.map(lambda x: f"{x:<,.2f}" if isinstance(x, float) else x),
            headers="keys",
            tablefmt="psql",
            showindex=False,
            stralign="right",
        )
        rich.print(formatted_table)
        # Save the result to a Excel file if requested
        if save:
            today = date.today().strftime("%Y%m%d")
            df.to_excel(
                f"view_{table}_{today}.xlsx", index=False, sheet_name="resources report"
            )
    else:
        logging.warning("Unable to show data. The query is none.")


@app.command()
def describe(
    table: Annotated[str, typer.Argument(help="Table name.")] = "project_resources",
):
    selected_table = TableName(table)
    articles = describer(table=selected_table)
    if articles is not None:
        with open(f"{table}.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{paragraph}\n" for paragraph in articles)
        rich.print(articles[0])
        rich.print("dan seterusnya...")


@app.command()
def validate(
    filename: Annotated[Optional[str], typer.Argument(help="File name.")] = None
) -> None:
    """
    Validate data from a file or run a full validation.

    Args:
        filename (str, optional): The name of the file to validate. Defaults to None.

    Returns:
        None

    Notes:
        This function validates data from a file if a filename is provided,
        or runs a full validation if no filename is provided.
    """
    if filename is None:
        run_validation()


def run_validation():
    project_resources = run_query(TableName.PROJECT_RESOURCES, output=4)
    engine = RuleEngine(project_resources=project_resources)
    results = engine.run()
    if results.empty:
        logging.info("No validation results.")
    else:
        logging.info("Saving validation results.")
        results = results.sort_values(
            by=["report_year", "wk_name", "field_name", "project_name", "uncert_lvl"]
        )
        results.to_csv(f"validation_{date.today().strftime('%Y%m%d')}.csv", index=False)


def load_esdc_data(
    filetype: FileType = FileType.CSV,
    to_file: bool = True,
    username: str = "",
    password: str = "",
) -> None:
    """
    Downloads and loads data from the ESDC API into the local database.

    This function downloads data from a specified URL and loads it into the ESDC database.
    If the corresponding file already exists, it can skip downloading and loading the data
    if the `to_file` parameter is set to `False`.

    Parameters
    ----------
    filetype : FileType
        The file type for the downloaded data. Currently, supports "csv" and "json".
    to_file : bool
        Whether to save the downloaded data to a file. Defaults to True.
    username : str
        The username for authenticating with the ESDC API.
    password : str
        The password for authenticating with the ESDC API.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Raises an exception if the download fails or if the data format is unsupported.
    """
    for table in TABLES:
        logging.info("Downloading %s table.", table.value)
        url = esdc_url_builder(table_name=table, file_type=filetype)
        logging.info("downloading from %s", url)
        data = esdc_downloader(url, username, password)
        if data is None:
            logging.warning("Failed to download %s data.", table.value)
            break
        if to_file:
            logging.debug("Save data as %s.%s", table.value, filetype.value)
            with open(table.value + "." + filetype.value, "wb") as f:
                f.write(data)

        if filetype == FileType.CSV:
            decoded_data = data.decode("utf-8").splitlines()
            # read_csv is used to handle remarks column that might
            # have commas in the string.
            content, header = _read_csv(decoded_data)
            load_data_to_db(content, header, table.value)
        elif filetype == FileType.JSON:
            parsed_json = json.loads(data)
            header = parsed_json[0].keys()
            content = [list(item.values()) for item in parsed_json]
            load_data_to_db(content, header, table.value)


def esdc_url_builder(
    table_name: TableName,
    api_ver: ApiVer = ApiVer.V2,
    verbose: int = 3,
    report_year: Optional[int] = None,
    file_type: FileType = FileType.CSV,
) -> str:
    """
    Build an ESDC URL based on the provided parameters.

    Parameters
    ----------
    table_name : TableName
        The table name to query.
    api_ver : ApiVer, optional
        The API version to use. Defaults to ApiVer.V2.
    verbose : int, optional
        The verbosity level. Defaults to 3.
    report_year : int, optional
        The report year to filter by. Defaults to None.
    file_type : FileType, optional
        The file type to request. Defaults to FileType.CSV.

    Returns
    -------
    str
        The constructed ESDC URL.

    Notes
    -----
    The URL is constructed by concatenating the base ESDC URL, API version,
    table name, verbosity level, and file type.
    If a report year is provided, it is added as a query parameter.
    For example, the url for project_resources table is:
    https://esdc.skkmigas.go.id/api/v2/project-resources?verbose=3&output=csv
    """
    load_dotenv(find_dotenv())
    url = os.getenv("ESDC_URL")
    if url is None:
        logging.info(
            "Environment Variables is not found. Url set to: https://esdc.skkmigas.go.id/"
        )
        url = Config.BASE_API_URL_V2

    url += api_ver.value
    tables = {
        TableName.PROJECT_RESOURCES: "project-resources",
        TableName.PROJECT_TIMESERIES: "project-timeseries",
    }
    url = f"{url}/{tables[table_name]}?verbose={verbose}"

    # TODO this is temporary fix since as of 2024-07-06
    # the API for time series does not support all year selection
    # remove this conditional if the API for project_timeseries is fixed.
    if table_name == TableName.PROJECT_TIMESERIES:
        report_year = 2023
    if report_year is not None:
        url = f"{url}&report-year={report_year}"
    url = f"{url}&output={file_type.value}"

    return url


def _load_file_as_csv(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    decoded_data = data.decode("utf-8").splitlines()
    content, header = _read_csv(decoded_data)
    load_data_to_db(content, header, table_name)


def _load_file_as_json(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    parsed_json = json.loads(data)
    header = parsed_json[0].keys()
    content = [list(item.values()) for item in parsed_json]
    load_data_to_db(content, header, table_name)


def esdc_downloader(
    url: str, username: str = "", password: str = ""
) -> Union[bytes, None]:
    """
    Download a file from a URL using the requests library and return its content.

    Parameters
    ----------
    url : str
        The URL of the file to download.

    Returns
    -------
    Union[bytes, None]
        The content of the downloaded file as bytes, or None if the download failed.

    Raises
    ------
    requests.exceptions.RequestException
        If there is a request error while downloading the file.

    """
    try:
        logging.info("requesting data to server...")
        response = requests.get(
            url, auth=(username, password), stream=True, timeout=300, verify=False
        )

        if response.status_code == 200:
            file_size = int(response.headers.get("Content-Length", 0))
            logging.debug("File size is %s bytes", file_size)
            logging.debug("Encoding format: %s", response.headers.get("Content-Encoding"))

            with closing(io.BytesIO()) as f:
                with Progress(
                    *Progress.get_default_columns(),
                    TransferSpeedColumn(),
                    DownloadColumn(binary_units=True),
                ) as progress:
                    task_id = progress.add_task(
                        f"[cyan]Downloading {round(file_size/1E6)} MB...",
                        total=file_size,
                        unit="B",
                        transfer=True,
                        speed_unit="B/s",
                    )
                    if response.headers.get("Content-Encoding") == "gzip":
                        with gzip.GzipFile(fileobj=response.raw, mode="rb") as gz:
                            while True:
                                chunk = gz.read(size=8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                                progress.update(task_id, advance=len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))

                    logging.info(
                        "File downloaded successfully to memory (Size: %s bytes)",
                        f.tell(),
                    )
                    return f.getvalue()

        logging.warning("File download failed. Status code: %s", response.status_code)
        return None
    except requests.exceptions.RequestException as e:
        logging.error("Request error while downloading the file. Error: %s", str(e))
        return None


def _read_csv(file: Union[str, Iterable]) -> Tuple[List[List[str]], List[str]]:
    """
    Reads a CSV file and returns its contents as a tuple of two values:
    a list of lists of strings representing the data,
    and a list of strings representing the header.

    Args:
        file: The path to the CSV file as a string,
        or an iterable of strings representing the CSV data.

    Returns:
        Tuple[List[List[str]], List[str]]: A tuple containing the data
        and header of the CSV file.
    """

    class _EsdcDialect(csv.Dialect):
        delimiter = ";"
        quotechar = '"'
        doublequote = True
        lineterminator = "\n"
        quoting = csv.QUOTE_STRINGS

    csv.register_dialect("esdc", _EsdcDialect)

    if isinstance(file, str):
        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, dialect="esdc")
    else:
        reader = csv.reader(file, dialect="esdc")
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)
    return data, header


if __name__ == "__main__":
    app()
