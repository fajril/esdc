import logging
from typing import List

import ollama

from esdc.selection import TableName
from esdc.dbmanager import run_query


def describer(table: TableName) -> List[str] | None:
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
        df = run_query(table=table, output=4)
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

            if row.project_class == "1. Reserves & Recoverables":
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
                    f" Potensi {prj_class} untuk {row.project_stage[3:]} secara nasional"
                    f" minyak sebesar {round(row.resources_oc / 1000)} MMSTB (juta barel)"
                    f" dan gas sebesar {round(row.resources_an)} BSCF (milyar kaki kubik)."
                )
            else:
                paragraph = (
                    f"{paragraph}"
                    f" Potensi {prj_class} untuk {row.project_stage[3:]} secara nasional"
                    f" minyak sebesar {round(row.resources_oc / 1000)} MMSTB (juta barel)"
                    f" dan gas sebesar {round(row.resources_an)} BSCF (milyar kaki kubik)."
                )

        elif table == TableName.FIELD_RESOURCES:

            if row.project_class == "1. Reserves & Recoverables":
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
                        f"{paragraph} memiliki potensi {prj_class} untuk {row.project_stage[3:]}"
                        f" minyak sebesar {round(row.rec_oc_risked)} MSTB dan gas sebesar {round(row.rec_an_risked)} BSCF."
                        f" Total proyek dalam klasifikasi {prj_class} untuk {row.project_stage[3:]} sebanyak {row.project_count} proyek."
                    )
                else:
                    paragraph = f"{paragraph} tidak memiliki potensi {prj_class} untuk {row.project_stage[3:]}."
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
        article.append(paragraph)
    return article


def exec_summarizer(
    text: str = "", data: str = "", context: str = "", model: str = "qwen2"
) -> str:
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
        prompt=f"""
                Write executive summary for this: 
                {text}

                use this information to write on DATA section: 
                {data}
                """,
        stream=False,
        options={"temperature": 0},
    )
    return response["response"]
