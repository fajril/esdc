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
from tqdm import tqdm
import click
import pandas as pd

from .selection import TableName, ApiVer, FileType
from .validate import RuleEngine


@click.command()
@click.option(
    "--ext",
    "-e",
    default="csv",
    type=str,
    help="Format of the downloaded file. Options: csv",
)
@click.option("--update", "-u", is_flag=True, help="Update the database.")
@click.option("--reload", "-r", is_flag=True, help="Reload db from bin files.")
@click.option("--validate", is_flag=True, help="Run Validation Engine")
@click.option("--file", default="", type=str)
@click.option("--verbose", "-v", count=True, help="Set log message level.")
@click.option("--output", "-o", count=True, help="Level of detail in the output data.")
@click.option("--save", "-s", is_flag=True, help="Write query result to file.")
@click.option("--show", default=None)
@click.option("--like", default="", type=str)
@click.option("--year")
@click.option("--columns", type=str)
def main(
    ext,
    update,
    reload,
    validate,
    file,
    verbose,
    output,
    save,
    show,
    like,
    year,
    columns,
):
    load_dotenv(find_dotenv())
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    logger.info("Log level set to %s", logging.getLevelName(logger.getEffectiveLevel()))

    if validate:
        run_validation()

    if update is not None:
        load_esdc_data(update=update, reload=reload, ext=FileType(ext), to_file=True)
    if show is not None:
        df = run_query(
            table=TableName(show),
            like=like,
            year=year,
            output=output,
            save=save,
            columns=columns,
        )
        pd.options.display.float_format = "{:,.2f}".format
        print(df)


def run_query(
    table: TableName,
    like: Optional[str] = None,
    year: Optional[str] = None,
    output: Optional[int] = 0,
    save: bool = False,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:

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
        pattern = r".*?(?=world)"
        query = re.sub(pattern, "", query)
        select_query = "SELECT"
        for col in columns:
            select_query += f" {col},"
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
    with sqlite3.connect("esdc.db") as conn:
        df = pd.read_sql_query(query, conn)

    if save:
        today = date.today().strftime("%Y%m%d")
        df.to_csv(f"view_{table.value}_{today}.csv")
    return df


def run_validation():
    project_resources = run_query(TableName.PROJECT_RESOURCES, output=4)
    engine = RuleEngine(project_resources=project_resources)
    results = engine.run()
    if results.empty:
        logging.info("Validation results: All checks have passed.")
    else:
        results.to_csv("validation.csv", index=False)


def load_esdc_data(
    update: bool = False,
    reload: bool = False,
    ext: FileType = FileType.CSV,
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
    tables = (TableName.PROJECT_RESOURCES, TableName.PROJECT_TIMESERIES)
    if update:
        for table in tables:
            logging.info("Downloading %s table.", table.value)
            url = esdc_url_builder(table_name=table)
            logging.info("downloading from %s", url)
            data = esdc_downloader(url)
            if data is None:
                logging.warning("Failed to download %s data.", table.value)
                break
            if to_file:
                logging.debug("Save data as %s .bin", table.value)
                with open(table.value + ".bin", "wb") as f:
                    f.write(data)
            if reload:
                if ext == FileType.CSV:
                    data = data.decode("utf-8").splitlines()
                    # read_csv is used to handle remarks column that might
                    # have commas in the string.
                    data, header = _read_csv(data)
                db_data_loader(data, header, table.value)
    if reload:
        for table in tables:
            if Path(table.value + ".bin").exists():
                _load_bin_as_csv(table.value + ".bin", table.value)
            else:
                logging.warning("File %s.bin is not found.", table.value)


def esdc_url_builder(
    table_name: TableName,
    api_ver: ApiVer = ApiVer.V2,
    verbose: int = 3,
    report_year: int = None,
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
        logging.error("Could not find Environment Variables.")
        raise FileNotFoundError()

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
    load_dotenv(find_dotenv())
    username = os.getenv("ESDC_USER")
    password = os.getenv("ESDC_PASS")

    try:
        response = requests.get(
            url, auth=(username, password), stream=True, timeout=300, verify=False
        )
        if response.status_code == 200:
            file_size = int(response.headers.get("Content-Length", 0))
            logging.debug("File size is %s bytes", file_size)

            with closing(io.BytesIO()) as f, tqdm(
                unit="B", total=file_size, unit_scale=True, leave=True
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                logging.info(
                    "File downloaded successfully to memory (Size: %s bytes)", f.tell()
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
        If there is an error while connecting to the database or executing a query.

    """
    create_table_query = {
        "project_resources": "create_table_project_resources.sql",
        "project_timeseries": "create_table_project_timeseries.sql",
    }
    logging.info("Connecting to the database.")
    with sqlite3.connect("esdc.db") as conn:
        cursor = conn.cursor()
        logging.debug("creating table %s in database", table_name)
        cursor.executescript(_load_sql_script(create_table_query[table_name]))
        column_names = ", ".join(["?" for _ in header])
        logging.debug("Inserting table data %s into the database.", table_name)
        insert_stmt = (
            f'INSERT INTO {table_name} ({", ".join(header)}) VALUES ({column_names})'
        )
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
            except Exception as e:
                logging.debug("insert statement: %s", insert_stmt)
                logging.debug("row data: %s", row)
                raise Exception(e)

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
    a list of lists of strings representing the data, and a list of strings representing the header.

    Args:
        file: The path to the CSV file as a string,
        or an iterable of strings representing the CSV data.

    Returns:
        Tuple[List[List[str]], List[str]]: A tuple containing the data and header of the CSV file.
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
