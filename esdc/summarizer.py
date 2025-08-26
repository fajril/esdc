import logging
from typing import List, Optional
from datetime import date

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

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
            if row.basin128 != "":
                paragraph = (
                    f"{paragraph}"
                    f" Lapangan ini berada di cekungan migas {row.basin128}."
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


def exec_summarizer(
    text: str = "", data: str = "", context: str = "", model: str = "gpt-4.1-mini"
) -> str | None:
    # Load environment variables
    load_dotenv(find_dotenv())

    client = OpenAI()

    sys_prompt = f"""
    Anda adalah seorang ahli teknik perminyakan senior yang bertugas membuat ringkasan eksekutif
    untuk pengambilan keputusan manajemen tingkat atas. Fokus pada wawasan bisnis penting dan
    metrik utama. Seluruh output harus dalam Bahasa Indonesia yang formal dan profesional.

    PERAN DAN AUDIENS:
    - Audiens utama: Eksekutif C-level dan manajemen senior
    - Tujuan: Memungkinkan pengambilan keputusan strategis yang tepat
    - Fokus: Metrik utama, tren, risiko, dan peluang

    KONTEKS DAN TERMINOLOGI:
                {context}

    PERSYARATAN ANALISIS:
    1. Analisis Kuantitatif:
       - Sajikan data numerik penting dengan satuan yang tepat
       - Sorot perubahan year-on-year
       - Tekankan deviasi atau tren yang signifikan

    2. Klasifikasi Sumber Daya:
       - Bedakan dengan jelas antara cadangan, sumber daya kontingen, dan sumber daya prospektif
       - Ikuti sistem klasifikasi SPE-PRMS
       - Sorot perubahan dalam kategorisasi sumber daya

    3. Penilaian Risiko:
       - Identifikasi risiko teknis dan operasional utama
       - Catat ketidakpastian geologis atau produksi
       - Tandai area yang memerlukan perhatian manajemen

    FORMAT OUTPUT:
    RINGKASAN EKSEKUTIF
    - Ikhtisar singkat temuan utama (2-3 kalimat)
    - Sorot perubahan atau perkembangan paling signifikan
    - Nyatakan poin tindakan segera jika ada

    CADANGAN (2P: Terbukti + Probable)
    - Posisi cadangan saat ini
    - Perubahan dari periode pelaporan sebelumnya
    - Faktor perolehan dan tingkat produksi
    - Tantangan atau peluang teknis utama

    SUMBER DAYA KONTINGEN
    - Volume sumber daya berdasarkan kategori utama
    - Status pengembangan dan milestone penting
    - Pertimbangan komersial
    - Persyaratan teknis untuk pematangan sumber daya

    SUMBER DAYA PROSPEKTIF
    - Potensi eksplorasi
    - Ringkasan penilaian risiko
    - Karakteristik prospek utama
    - Langkah selanjutnya yang direkomendasikan

    KESIMPULAN
    - Implikasi strategis
    - Tindakan yang direkomendasikan
    - Faktor-faktor penentu keberhasilan
    - Pertimbangan timeline

    PEDOMAN PENTING:
    1. Gunakan bahasa teknis yang tepat namun tetap jelas
    2. Kuantifikasi semua observasi jika memungkinkan
    3. Prioritaskan informasi material daripada detail minor
    4. Berikan konteks untuk semua angka signifikan
    5. Gunakan satuan standar industri secara konsisten
    6. Sorot poin keputusan dan rekomendasi dengan jelas
    7. Gunakan Bahasa Indonesia formal dan profesional
    8. Pertahankan istilah teknis dalam Bahasa Inggris jika diperlukan untuk kejelasan

    Catatan khusus untuk terminologi:
    - "Reserves" = Cadangan
    - "Resources" = Sumber Daya
    - "Contingent Resources" = Sumber Daya Kontingen
    - "Prospective Resources" = Sumber Daya Prospektif
    - "MMSTB" = Juta Barel
    - "BSCF" = Milyar Kaki Kubik
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"""
                Buat ringkasan eksekutif untuk informasi ini:
                {text}

                Gunakan informasi ini untuk bagian DATA:
                {data}
                """}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def analyzer(field_name, wk_name, base_url=None, model="gpt-4.1-mini") -> str | None:
    """
    """
    try:
        with open("sys_prompt.md", "r", encoding="utf-8") as file:
            sys_prompt = file.read()
    except FileNotFoundError:
        logging.error("Error: sys_prompt.md file not found")
    except Exception as e:
        logging.error(f"Error reading file: {e}")

    df = run_query(table=TableName.PROJECT_RESOURCES, output=4)
    if df is not None:
        df = df[
            (df['field_name'].str.contains(field_name, case=False)) &
            (df['wk_name'].str.contains(wk_name, case=False)) &
            (df['uncert_lvl'].str.contains("Middle", case=False)) &
            (df["report_year"] == 2024)
        ]
        paragraph = (
            f"Tanggal hari ini adalah {date.today()}. "
            f"Pada lapangan {df['field_name'].iat[0]} "
            f"di Wilayah Kerja {df['wk_name'].iat[0]}. "
            f"Kontrak Wilayah kerja ini berakhir di {df['psc_eff_end'].iat[0]}. "
            f"Lapangan ini berada di latitude {df['field_lat'].iat[0]} dan longitude {df['field_long'].iat[0]}"
            f"Pada status pelaporan 31.12.{df['report_year'].iat[0]} lapangan ini memiliki project berikut: \n\n"
        )

        project_details = []
        for row in df.itertuples():
            if row.onstream_year >= row.report_year:
                filler_word = f"Proyek ini berencana untuk onstream di tahun {row.onstream_year}. "
            else:
                filler_word = (
                    f"Proyek ini telah berproduksi sejak {row.onstream_actual}. "
                    "Total produksi kumulatif dari awal onstream"
                    f" hingga hingga 31.12.{row.report_year} sebesar "
                    f"{row.cprd_sls_oc} MSTB dan {row.cprd_sls_an} BSCF. "
                )
            if row.prj_ioip + row.prj_igip > 0:
                filler_ipip = (
                    f"Proyek ini memiliki IOIP P50 {row.prj_ioip} MSTB. dan IGIP P50 {row.prj_igip} BSCF. "
                )
            else:
                filler_ipip = "Proyek ini tidak memberikan tambahan in place baru."
            project_detail = (
                f"Proyek {row.project_name} "
                f"memiliki project maturity level {row.project_level}. "
                f"Mekanisme produksi dari proyek ini adalah {row.prod_stage}. "
                f"Proyek ini memiliki resources {row.rec_oc} MSTB dan {row.rec_an} BSCF. "
                f"{filler_ipip}"
                "Apabila angka resources pada catatan berbeda dengan angka ini, maka gunakan angka ini."
                f"{filler_word}"
                f"pada tahun {row.report_year} "
                f"proyek ini berproduksi sebanyak {row.rate_sls_oc} MSTB dan {row.rate_sls_an} BSCF. "
                f"catatan dalam proyek ini adalah {row.project_remarks}"
            )
            project_details.append(project_detail)

        paragraph += "".join(project_details)
    else:
        paragraph = "Data is not accessible. Check your data."

    load_dotenv(find_dotenv())
    client = OpenAI(base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", 
             "content": f"Berikut adalah data yang perlu dianalisa: {paragraph}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


