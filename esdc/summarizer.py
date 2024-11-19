import logging
from typing import List, Optional

import ollama

from esdc.selection import TableName
from esdc.dbmanager import run_query


def describer(table: TableName, year: Optional[int], search: Optional[str]) -> List[str] | None:
    if table == TableName.NKRI_RESOURCES:
        df = run_query(table=table)
        if df is not None:
            df.sort_values(
                by=["report_year", "project_class"],
                ascending=[False, True],
                inplace=True,
            )
        else:
            logging.warning("No query result for %s.", table.value)
            return None
    else:
        df = run_query(table=table, like=search, year=year, output=4)
        if df is not None:
            df.sort_values(
                by=["report_year", "wk_name", "field_name", "project_class"],
                ascending=[False, True, True, True],
                inplace=True,
            )
        else:
            logging.warning("No query result for %s.", table.value)
            return None
    article = []
    df = df[df.uncert_lvl == "2. Middle Value"]

    for _, row in df.iterrows():
        if table == TableName.NKRI_RESOURCES:

            if row.project_class == "1. Reserves & GRR":
                prj_class = "Government of Indonesia Recoverable Resources (GRR)"
            else:
                prj_class = row.project_class[3:]
            paragraph = f"Berdasarkan laporan status 31.12.{row.report_year},"
            if prj_class == "Government of Indonesia Recoverable Resources (GRR)":
                paragraph = (
                    f"{paragraph}"
                    f" cadangan nasional 2P (proven + probable reserves)"
                    f" minyak sebesar {round(row.reserves_oc / 1000)} MMSTB (juta barel)"
                    f" dan gas sebesar {round(row.reserves_an)} BSCF (milyar kaki kubik)."
                    f" Potensi {prj_class} untuk {row.project_stage} secara nasional"
                    f" minyak sebesar {round(row.resources_oc / 1000)} MMSTB (juta barel)"
                    f" dan gas sebesar {round(row.resources_an)} BSCF (milyar kaki kubik)."
                )
            else:
                paragraph = (
                    f"{paragraph}"
                    f" Potensi {prj_class} untuk {row.project_stage} secara nasional"
                    f" minyak sebesar {round(row.resources_oc / 1000)} MMSTB (juta barel)"
                    f" dan gas sebesar {round(row.resources_an)} BSCF (milyar kaki kubik)."
                )

        elif table == TableName.FIELD_RESOURCES:

            if row.project_class == "1. Reserves & GRR":
                prj_class = "Government of Indonesia Recoverable Resources (GRR)"
            else:
                prj_class = row.project_class[3:]

            paragraph = (
                f"Berdasarkan laporan status 31.12.{row.report_year}, lapangan {row.field_name}"
                f" (field id: {row.field_id} lat: {row.field_lat} long: {row.field_long})"
                f" berada di wilayah kerja {row.wk_name} dengan operator {row.operator_name}."
            )
            if row.basin86 != "":
                paragraph = (
                    f"{paragraph}"
                    f" Lapangan ini berada di cekungan migas {row.basin86}."
                    f" Lapangan ini"
                )
            else:
                paragraph = f"{paragraph}" f" Lapangan ini"
            if prj_class == "Government of Indonesia Recoverable Resources (GRR)":
                paragraph = (
                    f"{paragraph}"
                    f" memiliki cadangan 2P (proven + probable reserves) minyak sebesar {round(row.res_oc)} MSTB"
                    f" dan gas sebesar {round(row.res_an)} BSCF."
                    f" Lapangan ini juga memiliki potensi {prj_class}"
                    f" minyak sebesar {round(row.rec_oc_risked)} MSTB dan gas sebesar {round(row.rec_an_risked)} BSCF."
                )
            else:
                if row.rec_oc_risked + row.rec_an_risked > 0:
                    paragraph = (
                        f"{paragraph} memiliki potensi {prj_class} untuk {row.project_stage}"
                        f" minyak sebesar {round(row.rec_oc_risked)} MSTB dan gas sebesar {round(row.rec_an_risked)} BSCF."
                        f" Total proyek dalam klasifikasi {prj_class} untuk {row.project_stage} sebanyak {row.project_count} proyek."
                    )
                else:
                    paragraph = f"{paragraph} tidak memiliki potensi {prj_class} untuk {row.project_stage}."
            paragraph = (
                f"{paragraph}"
                f" Volume Initial Oil in Place (IOIP) sebesar {round(row.ioip)} MSTB dan"
                f" volume Initial Gas in Place (IGIP) sebesar {round(row.igip)} BSCF."
            )
            if row.cprd_sls_oc + row.cprd_sls_an > 0:
                paragraph = (
                    f"{paragraph}"
                    f" Produksi kumulatif minyak sebesar {round(row.cprd_sls_oc)} MSTB,"
                    f" sedangkan gas sebesar {round(row.cprd_sls_an)} BSCF."
                )
            if row.field_remarks != "":
                paragraph = f"{paragraph} Catatan dari tiap proyek sebagai berikut: {row.field_remarks}"
        else:
            paragraph = f"Deskripsi dari {table} belum tersedia."
        article.append(paragraph)
    return article

def create_chat_session(model: str = "qwen2.5:latest", system_prompt: Optional[str] = None):
    """
    Create a chat session using the ollama module.
    
    Args:
    model (str): The name of the model to use. Defaults to "qwen2.5:latest".
    system_prompt (Optional[str]): An optional system prompt to set the context for the chat.
    
    Returns:
    A function that can be used to send messages to the chat session.
    """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        def chat_function(user_message: str):
            nonlocal messages
            messages.append({"role": "user", "content": user_message})
            response = ollama.chat(model=model, messages=messages)
            assistant_message = response['message']['content']
            messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        return chat_function

    except Exception as e:
        logging.error(f"Error creating chat session: {str(e)}")
        return None

