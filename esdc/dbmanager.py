import sqlite3
import logging
import re
from pathlib import Path
from typing import Union, List, Optional

import pandas as pd

from esdc.configs import Config
from esdc.selection import TableName


def load_data_to_db(
    content: List[List[str]], header: List[str], table_name: str
) -> None:
    """
    Load data into the ESDC database.

    Parameters
    ----------
    content : str
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
    if not Config.get_db_path().exists():
        Config.get_db_path().mkdir(parents=True, exist_ok=True)
        logging.info("Database does not exist. Creating new database.")
        logging.debug("Database location: %s", Config.get_db_path())
    with sqlite3.connect(Config.get_db_path() / "esdc.db") as conn:
        cursor = conn.cursor()
        logging.debug("creating table %s in database", table_name)
        cursor.executescript(_load_sql_script(create_table_query[table_name]))
        column_names = ", ".join(["?" for _ in header])
        insert_stmt = (
            f'INSERT INTO {table_name} ({", ".join(header)}) VALUES ({column_names})'
        )
        logging.debug("Inserting table data %s into the database.", table_name)
        try:
            cursor.executemany(insert_stmt, content)
        except sqlite3.Error as e:
            logging.debug("insert statement: %s", insert_stmt)
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


def run_query(
    table: TableName,
    like: Optional[str] = None,
    year: Optional[int] = None,
    output: int = 0,
    columns: Union[str, List[str]] = "",
) -> pd.DataFrame | None:

    output = min(output, 4)

    sql_script_map = {
        TableName.PROJECT_RESOURCES: "view_project_resources.sql",
        TableName.FIELD_RESOURCES: "view_field_resources.sql",
        TableName.WA_RESOURCES: "view_wa_resources.sql",
        TableName.NKRI_RESOURCES: "view_nkri_resources.sql",
    }
    queries = _load_sql_script(sql_script_map[table])

    # Extract the query from the SQL script
    query = queries.split(";")[max(0, output - 1)].strip()

    # Modify the query based on the provided columns
    if columns:
        logging.debug("selected columns: %s", columns)
        # Define the regex pattern to match any characters (including none)
        # that come before the string "FROM" in a non-greedy way.
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
        with sqlite3.connect(Config.get_db_path() / "esdc.db") as conn:
            df = pd.read_sql_query(query, conn)
    except sqlite3.OperationalError:
        logging.error(
            """Cannot query data. Database file does not exist.
        Try to run this command first:
        
        esdc fetch --save  
        """
        )
        return None

    return df


def _load_sql_script(script_file: str) -> str:
    file = Path(__file__).parent / "sql" / script_file
    with open(file, "r", encoding="utf-8") as f:
        return f.read()
