import os
import io
from pathlib import Path
import csv
import sqlite3
from contextlib import closing
import logging
from typing import Union, List, Tuple, Iterable

import requests
from dotenv import load_dotenv
from tqdm import tqdm
import click

from .selection import TableName, ApiVer, FileType

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)

click.command()
click.option(
    "--ext",
    "-e",
    default="csv",
    type=str,
    help="Format of the downloaded file. Options: csv",
)
click.option("--update", "-u", default=True, type=bool, help="Update the database.")


def _cli(ext: str = "csv", update: bool = True) -> None:
    load_esdc_data(update=update, ext=FileType(ext), to_file=True)


def load_esdc_data(
    update: bool = False, ext: FileType = FileType.CSV, to_file=True
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
    if update:
        for table, url in TableName:
            logging.info("Downloading %s table.", table.value)
            url = esdc_url_builder(table_name=table)
            data = esdc_downloader(url)
            if data is None:
                logging.warning("Failed to download %s data.", table.value)
                break
            if to_file:
                logging.debug("Save data as %s.bin", table.value)
                with open(table.value + ".bin", "wb") as f:
                    f.write(data)
            if ext == "csv":
                data = data.decode("utf-8").splitlines()
                # read_csv is used to handle remarks column that might
                # have commas in the string.
                data, header = _read_csv(data)
            db_data_loader(data, header, table.value)
    else:
        for table in TableName:
            if Path(table.value + ".bin").exists():
                _load_bin_as_csv(table.value + ".bin", table.value)
            else:
                logging.warning("File %s.bin is not found.")


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
    """

    url = os.getenv("ESDC_URL")

    url += api_ver.value
    url += f"/{table_name.value}?"
    url += f"verbose={verbose}"
    if report_year is not None:
        url += f"&report_year={report_year}"
    url += f"output={file_type.value}"

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
            cursor.execute(insert_stmt, row)
        logging.debug("Creating uuid column for table %s", table_name)
        cursor.executescript(_create_table_uuid(table_name))

        if table_name == "project_resources":
            logging.debug("creating project_stage column in project_resources")
            cursor.executescript(
                _load_sql_script("create_project_resources_project_stage.sql")
            )

            logging.debug("Creating table view for field, working area, nkri.")
            cursor.executescript(_load_sql_script("create_esdc_view.sql"))
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

    if isinstance(file, list):
        reader = csv.reader(file)
    if isinstance(file, str):
        with open(file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
    else:
        reader = csv.reader(file)
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)
    return data, header


def _create_table_uuid(table_name: str):
    return f"""
    -- add new column to store project_uuid
    ALTER TABLE {table_name}
    ADD COLUMN uuid TEXT;

    -- generate UUID v4 in {table_name} table. UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    UPDATE {table_name}
    SET uuid = (
        lower(hex(randomblob(4))) || '-' ||
        lower(hex(randomblob(2))) || '-' ||
        '4' || substr(hex(randomblob(2)), 2) || '-' ||
        substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
        lower(hex(randomblob(6)))
    );
    """


def _load_sql_script(script_file: str) -> str:
    file = Path(__file__).parent / "sql" / script_file
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    _cli()
