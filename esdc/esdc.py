import os
import io
from datetime import date
from pathlib import Path
import csv
import sqlite3
from contextlib import closing
import logging
import re
from typing import Union, List, Tuple, Iterable, Optional

import requests
from dotenv import load_dotenv, find_dotenv
import typer
from typing_extensions import Annotated
import rich
from rich.progress import Progress
from rich.logging import RichHandler
from rich.prompt import Prompt
from platformdirs import PlatformDirs
import pandas as pd

from .selection import TableName, ApiVer, FileType
from .validate import RuleEngine

APP_NAME = "esdc"
APP_AUTHOR = "skk"
dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR)
DB_PATH = dirs.user_data_path
BASE_API_URL_V2 = "https://esdc.skkmigas.go.id/api/v2"
TABLES: Tuple[str] = (TableName.PROJECT_RESOURCES, TableName.PROJECT_TIMESERIES)

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
    handler = RichHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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
def init():
    fetch()


@app.command()
def fetch(
    filetype: str = typer.Option("csv", help="Options: csv, json, zip"),
    save: bool = typer.Option(False, "--save/--no-save", help="Save to file or not"),
) -> None:
    """
    Fetch data from ESDC and save it to a file.

    Args:
        filetype (str, optional): The type of file to save the data to. Defaults to "csv".
        save (bool, optional): Whether to save the data to a file. Defaults to True.

    Returns:
        None

    Notes:
        This function fetches data from ESDC and saves it to a file based on the provided filetype.
        If the filetype is not available, a log message will be printed.
    """
    if filetype == "csv":
        load_esdc_data(filetype=FileType.CSV, to_file=save)
    elif filetype == "json":
        load_esdc_data(filetype=FileType.JSON, to_file=save)
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
        if Path(table.value + "." + filetype.value).exists():
            filename = table.value + ".bin"
            if filetype == "csv":
                _load_bin_as_csv(filename, table.value)
            else:
                logging.debug(
                    "failed to load %s. Unknown %s format", filename, filetype
                )
        else:
            logging.warning("File %s is not found.", filename)


@app.command()
def show(
    table: Annotated[str, typer.Argument(help="Table name.")],
    like: Annotated[Optional[str], typer.Option(help="Filter value")] = "",
    year: Annotated[
        Optional[int], typer.Option(min=2019, help="Filter year value")
    ] = None,
    output: Annotated[Optional[int], typer.Option(help="Detail of output.")] = 0,
    save: Annotated[Optional[bool], typer.Option(help="Save the output data")] = True,
    columns: Annotated[Optional[str], typer.Option(help="select column")] = None,
):
    """
    Show data from a specific table.

    Args:
        table (str): The name of the table to show data from.
        like (str, optional): A filter value to apply to the data. Defaults to "".
        year (str, optional): A filter year value to apply to the data. Defaults to None.
        output (int, optional): The level of detail to show in the output. Defaults to 0.
        save (bool, optional): Whether to save the output data. Defaults to True.

    Returns:
        None

    Notes:
        This function runs a query on the specified table with the provided filters
        and displays the result.
        The output is formatted to display float values with two decimal places.
        If the save option is True, the output data will be saved.
    """
    if columns is not None:
        columns = columns.split(" ")
    df = run_query(
        table=TableName(table),
        like=like,
        year=year,
        output=output,
        save=save,
        columns=columns,
    )
    if df is not None:
        pd.options.display.float_format = "{:,.2f}".format
        rich.print(df.to_string(index=False))


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


def run_query(
    table: TableName,
    like: Optional[str] = None,
    year: Optional[int] = None,
    output: Optional[int] = 0,
    save: bool = False,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame | None:

    output = min(output, 4)

    if table == TableName.PROJECT_RESOURCES:
        queries = _load_sql_script("view_project_resources.sql")
    elif table == TableName.FIELD_RESOURCES:
        queries = _load_sql_script("view_field_resources.sql")
    elif table == TableName.WA_RESOURCES:
        queries = _load_sql_script("view_wa_resources.sql")
    else:
        queries = _load_sql_script("view_nkri_resources.sql")

    if output > 1:
        query = queries.split(";")[output - 1].strip()
    else:
        query = queries.split(";")[0].strip()

    if columns is not None:
        pattern = r".*?(?=FROM)"
        query = re.sub(pattern, "", query)
        select_query = "SELECT" + ", ".join(col for col in columns)
        logging.debug("query: %s", select_query)
        query = select_query[:-1] + query

    # Replace placeholders with actual values
    if table == TableName.NKRI_RESOURCES:
        if year is None:
            query = query.replace("WHERE report_year = {year}", "")
        else:
            query = query.replace("{year}", str(year))
    else:
        if like is None:
            query = query.replace("{like}", "")
        else:
            query = query.replace("{like}", like)

        if year is None:
            query = query.replace("AND report_year = {year}", "")
        else:
            query = query.replace("{year}", str(year))

    # Execute the query
    try:
        with sqlite3.connect(DB_PATH / "esdc.db") as conn:
            df = pd.read_sql_query(query, conn)
    except sqlite3.OperationalError:
        logging.error(
            """Cannot query data. Database file does not exist.
        Try to run this command first:
        
        esdc fetch --save  
        """
        )
        return None

    if save:
        today = date.today().strftime("%Y%m%d")
        df.to_csv(f"view_{table.value}_{today}.csv")
    return df


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
    to_file=True,
) -> None:
    """
    Downloads and loads data into the ESDC database.

    This function downloads data from a specified URL and loads it into the ESDC database.
    If the corresponding file already exists,
    it can skip downloading and loading the data if the `update` parameter is set to `False`.

    Parameters
    ----------
    ext : str
        The file extension for the downloaded data. Currently, only "csv" is supported.
    update : bool
        Whether to update the database or skip downloading and loading data
        if the corresponding file already exists.

    Returns
    -------
    None
        None

    Raises
    ------
    click.exceptions.BadParameter
        If an invalid file extension is provided.

    """
    for table in TABLES:
        logging.info("Downloading %s table.", table.value)
        url = esdc_url_builder(table_name=table, file_type=filetype)
        logging.info("downloading from %s", url)
        data = esdc_downloader(url)
        if data is None:
            logging.warning("Failed to download %s data.", table.value)
            break
        if to_file:
            logging.debug("Save data as %s.%s", table.value, filetype.value)
            with open(table.value + "." + filetype.value, "wb") as f:
                f.write(data)

        if filetype == FileType.CSV:
            data = data.decode("utf-8").splitlines()
            # read_csv is used to handle remarks column that might
            # have commas in the string.
            data, header = _read_csv(data)
            db_data_loader(data, header, table.value)


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
        logging.warning(
            "Could not find Environment Variables. Url set to: https://esdc.skkmigas.go.id/api/v2/"
        )
        url = BASE_API_URL_V2

    url += api_ver.value
    tables = {
        TableName.PROJECT_RESOURCES: "project-resources",
        TableName.PROJECT_TIMESERIES: "project-timeseries",
    }
    url += f"/{tables[table_name]}?"
    url += f"verbose={verbose}"
    # TODO this is temporary fix since as of 2024-07-06
    # the API for time series does not support all year selection
    # remove this conditional if the API for project_timeseries is fixed.
    if table_name == TableName.PROJECT_TIMESERIES:
        report_year = 2023
    if report_year is not None:
        url += f"&report-year={report_year}"
    url += f"&output={file_type.value}"

    return url


def _load_bin_as_csv(file: str, table_name):
    with open(file, "rb") as f:
        data = f.read()
    data = data.decode("utf-8").splitlines()
    content, header = _read_csv(data)
    db_data_loader(content, header, table_name)


def esdc_downloader(url: str) -> Union[bytes, None]:
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
    username: str = ""
    password: str = ""

    load_dotenv(find_dotenv())
    username = os.getenv("ESDC_USER")
    password = os.getenv("ESDC_PASS")

    if not username:
        username = Prompt("user: ")
        password = Prompt("pass: ", password=True)

    try:
        response = requests.get(
            url, auth=(username, password), stream=True, timeout=300, verify=False
        )

        if response.status_code == 200:
            file_size = int(response.headers.get("Content-Length", 0))
            logging.debug("File size is %s bytes", file_size)

            with closing(io.BytesIO()) as f:
                with Progress() as progress:
                    task_id = progress.add_task(
                        f"[red]Downloading {round(file_size/1E6)} MB...",
                        total=file_size,
                        unit="B",
                        transfer=True,
                        speed_unit="B/s",
                    )
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


def db_data_loader(data: List, header: List[str], table_name: str) -> None:
    """
    Load data into the ESDC database.

    Parameters
    ----------
    data : str
        The data to load into the database as a string.
    table_name : str
        The name of the table to load the data into.

    Returns
    -------
    None

    Raises
    ------
    sqlite3.Error
        If there is an error while connecting
        to the database or executing a query.

    """
    create_table_query = {
        "project_resources": "create_table_project_resources.sql",
        "project_timeseries": "create_table_project_timeseries.sql",
    }
    logging.info("Connecting to the database.")
    if not DB_PATH.exists():
        DB_PATH.mkdir(parents=True, exist_ok=True)
        logging.info("Database does not exist. Creating new database.")
        logging.debug("Database location: %s", DB_PATH)
    with sqlite3.connect(DB_PATH / "esdc.db") as conn:
        cursor = conn.cursor()
        logging.debug("creating table %s in database", table_name)
        cursor.executescript(_load_sql_script(create_table_query[table_name]))
        column_names = ", ".join(["?" for _ in header])
        logging.debug("Inserting table data %s into the database.", table_name)
        insert_stmt = f'INSERT INTO {table_name} ({", ".join(header)}) VALUES ({
                column_names})'
        for i, row in enumerate(data):
            if len(row) != len(header):
                logging.debug(
                    "row %s, total data column: %s vs %s",
                    i,
                    len(row),
                    len(header),
                )
            try:
                cursor.execute(insert_stmt, row)
            except sqlite3.Error as e:
                logging.debug("insert statement: %s", insert_stmt)
                logging.debug("row data: %s", row)
                raise sqlite3.Error(str(e)) from e

        logging.debug("Creating uuid column for table %s", table_name)
        uuid_query = _load_sql_script("create_column_uuid.sql")
        cursor.executescript(uuid_query.replace("{table_name}", table_name))

        if table_name == "project_resources":
            logging.debug("creating project_stage column in project_resources")
            cursor.executescript(
                _load_sql_script("create_project_resources_project_stage.sql")
            )

            logging.debug("Creating table view for field, working area, nkri.")
            cursor.executescript(_load_sql_script("create_esdc_view.sql"))
        cursor.execute("VACUUM;")
        conn.commit()

        logging.info("Table %s is loaded into database.", table_name)


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

    class EsdcDialect(csv.Dialect):
        delimiter = ";"
        quotechar = '"'
        doublequote = True
        lineterminator = "\n"
        quoting = csv.QUOTE_STRINGS

    csv.register_dialect("esdc", EsdcDialect)

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


def _load_sql_script(script_file: str) -> str:
    file = Path(__file__).parent / "sql" / script_file
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app()
