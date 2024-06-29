import sqlite3
import pandas as pd
import logging

import ollama

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def query_table(table_name: str) -> pd.DataFrame:
    """
    Query a table from an SQLite database and return a Pandas DataFrame.

    Parameters
    ----------
    table_name : str
        The name of the table to query.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the queried table.

    Raises
    ------
    sqlite3.Error
        If there is an error connecting to the database or executing the query.
    """
    try:
        with sqlite3.connect("esdc.db") as conn:
            cursor = conn.cursor()
            cursor.execute(f"""SELECT * FROM {table_name}""")
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            df = pd.DataFrame(results, columns=columns)
    except sqlite3.Error as e:
        logging.error("Exception occured: %s", str(e))
        raise e

    return df

def exec_summarizer(text: str="", data:str="", context: str="", model: str="qwen2") -> str:
    SYSTEM_PROMPT = f"""
                Your role is a petroleum engineering expert. 
                Your task is to create report in the form of executive summary 
                from the given information.
                The executive summary is meant to upper management.
                They need all the necessary information to make decisions.
                the full capital text is a project name. for example:
                TANJUNG - BASE is a name of the project.
                Think step by step. here's the step you must follow:
                1. use the context to determine the abbreviation in information: 
                {context}
                2. add all the data from the context into the summary.
                3. decide whether the information is clear or need summarization.
                4. If it needs summarization, devise a plan to summarize it.
                5. If it does not need summarization, use the existing information.
                6. write the summary.
                The format of executive summary is:
                EXECUTIVE SUMMARY
                <<write brief introduction to the executive summary>>

                RESERVES
                <<Write all the data related to 1. Reserves & Recoverables here>>

                CONTINGENT RESOURCES
                <<Write all the data related to 2. Contingent Resources here>>

                PROSPECTIVE RESOURCES
                <<Write all the data related to 3. Prospective Resources here>>

                CONCLUSION
                <<Write the conclusion here>>
"""

    response = ollama.generate(
        model=model,
        system=SYSTEM_PROMPT,
        prompt= f"""
                Write executive summary for this: 
                {text}

                use this information to write on DATA section: 
                {data}
                """,
        stream=False,
        options={"temperature": 0}
    )
    return response["response"]