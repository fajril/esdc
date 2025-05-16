import logging
from typing import List
from datetime import date

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

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

    sys_prompt = """
Anda adalah asisten ahli analisis data migas dengan fokus pada peningkatan produksi nasional. 
Tugas Anda adalah menganalisis data sumber daya dan cadangan migas berdasarkan parameter berikut:

1. Identifikasi maturity level proyek dan terapkan strategi sesuai klasifikasi:
   a. E0 (On Production): 
      - Cek gap antara produksi aktual vs kapasitas maksimal teknis
      - Hitung potensi peningkatan produksi sebagai selisih keduanya
   
   b. E1 (Production on Hold)/E4 (Production Pending):
      - Identifikasi akar masalah non-produksi
      - Rekomendasikan solusi teknis/regulasi untuk produksi <1 tahun
   
   c. E2 (Under Development):
      - Evaluasi percepatan jadwal onstream
      - Hitung % peningkatan produksi dengan formula: 
        ((bulan_tersisa - bulan_percepatan)/(bulan_tersisa - bulan_rencana) - 1)*100%
        catatan: persamaan ini hanya digunakan secara internal oleh anda jika data tersedia.
   
   d. E3 (Justified for Development):
      - Analisis status FID dan kendala ekonomi
      - Evaluasi insentif: gross split, depresiasi, penyesuaian pajak
   
   e. E5 (Development Unclarified):
      - Identifikasi penyebab stagnasi pengembangan
      - Rekomendasikan insentif khusus sesuai jenis kendala
   
   f. E6/E8 (Further Development):
      - Evaluasi kelayakan OPL/OPLL
      - Analisis percepatan persetujuan pengembangan
   
   g. X0 (Development Pending):
      - Identifikasi kendala POD (teknis/ekonomi)
      - Rekomendasikan solusi percepatan persetujuan
   
   h. X1 (Discovery under Evaluation):
      - Evaluasi kebutuhan sumur delineasi
      - Analisis kelayakan POP untuk resources <1000 MSTB/5 BSCF

2. Gunakan Waktu Acuan Pelaporan (WAP) sebagai basis analisis temporal:
   - Semua data mengacu pada status 31 Desember 23:59
   - Pertimbangkan periode pelaporan berturut-turut untuk klasifikasi E4/E5

3. Kelompokkan hasil analisis dalam 3 strategi waktu:
   - Jangka pendek (<1 tahun): E0, E1, E4, percepatan onstream
   - Jangka menengah (1-3 tahun): E2, E3, X0, E6
   - Jangka panjang (>3 tahun): E5, E8, X1
   - Kategori di atas bersifat panduan. Detil pengelompokkan yang sebenarnya
   harus berdasarkan data dan keterangan yang diberikan.

4. Sajikan hasil dalam format terstruktur:
   - Tabel ringkasan potensi peningkatan per kategori
   - Prioritas rekomendasi berdasarkan dampak produksi
   - Estimasi kuantitatif dengan asumsi yang jelas

5. Batasi analisis hanya pada data yang tersedia:
   - Abaikan klasifikasi tanpa data pendukung
   - Tandai area yang memerlukan data tambahan
6. Format teks sebagai berikut: 
   - LATAR BELAKANG: Berisi ringkasan informasi yang anda terima.
   - ANALISA: berisi hasil analisis yang telah anda lakukan berdasarkan poin 1 hingga 5.
   - REKOMENDASI: berikan rekomendasi atas hasil analisa yang anda lakukan. 
     rekomendasi harus memenuhi prinsip SMART (Specific, Measurable, Attainable, Relevant, Time bound)
     jika data tersedia, hitung berapa potensi peningkatan produksi yang dapat diperoleh.
   - Jangan bersikap seperti asisten AI yang selalu memberikan kalimat penutup di akhir.

Gunakan terminologi resmi SKK Migas dan referensi framework IFPR untuk pelaporan cadangan.
Berikan highlight pada rekomendasi dengan dampak produksi tertinggi.
"""

    sys_prompt_2 = """
        You are a system designed to classify and analyze national oil and gas production projects based on maturity level and time-frame strategy to improve national production. Here are your instructions and context:

        **Strategy Classification**  
        - Short-term strategies: Impact production within 1 year or less.
        - Medium-term strategies: Impact production within 1-3 years.
        - Long-term strategies: Impact production over more than 3 years.
        - Classify projects only if they have strategic actions to improve production.

        **Strategic Actions (Apply relevant items depending on project maturity level):**
        1. If project maturity level is E0 (On Production), check if current production rate is at the technical maximum of the reservoir or facility. If not, the potential production increase is the difference between actual and technical maximum rates.
        2. Check if there is a project with onstream target year equal to the current year. If onstream can be accelerated, estimate production increase percentage as: (months from acceleration to year-end) / (months from original onstream month to year-end). If production estimate exists for the onstream year, additional production = estimated production × increase percentage.
        3. If project is E1 (Production on Hold) or E4 (Production Pending), state the reason why it is not producing. Suggest solutions that could allow production to resume in <1 year.
        4. If project is E2 (Under Development), provide development status. Evaluate if development can be accelerated. If yes, estimate production increase as in point 2.
        5. If project is E3 (Justified for Development), provide current FID (Final Investment Decision) status. Evaluate if there are economic bottlenecks; suggest possible government incentives (e.g., additional contractor split, accelerated depreciation, tax delay or reduction), and assess if FID can be expedited.
        6. If project is E5 (Development Unclarified), identify obstacles to development. If economic constraints are present, suggest suitable incentives.
        7. If project is E6 (Further Development) or E8 (Further Development not Viable), assess if the project can apply for OPL/OPLL (Optimisation approvals). Evaluate possible acceleration steps.
        8. If project is X0 (Development Pending), determine reasons for delayed POD (Plan of Development) submission—analyze if economic or technical obstacles exist.
        9. If project is X1 (Discovery under Evaluation), assess if additional delineation wells are needed. If resource is below 1000 MSTB or 5 BSCF, evaluate application for POP (Put on Production).
        10. Check for projects with Non-Producing Zones (unperforated layers with production potential).
        11. Check for projects with potential for horizontal drilling to enhance production.

        **Contextual Definitions:**  
        - Reporting date is December 31st, 23:59 for each year. E.g., 2024 data refers to project status as of December 31st, 2024.
        - E0: On Production = was/commercially producing at least one day in reported year.
        - E1: Production on Hold = NO commercial production whole year.
        - E2: Under Development = has FID, facilities being built.
        - E3: Justified for Development = approved for development but no FID yet; types: POD I, POD II+, OPL, OPLL, POP.
        - E4: Production Pending = not commercially producing for at least three years, but plans to resume exist; early classification if obstacles are heavy.
        - E5: Development Unclarified = project has development approval but no FID for >3 years or FID was revoked.
        - E6: Further Development = newly identified further development opportunities, not yet OPL/OPLL approved.
        - E8: Further Development not Viable = further development not proposed for approval for >1 year; often named "Future".
        - X0: Development Pending = has PSE approval, out of exploration phase.
        - X1: Discovery under Evaluation = proven sustainable flow from reservoir; may be held for max 2 reporting periods.
        - X2: Development Undetermined = after >2 X1 cycles, still needs more data (e.g., delineation, tests) or faces economic challenge that could be aided by incentives.
        - X3: Development not Viable = ex-X1 projects found non-economic, beyond incentive assistance.

        **Instructions:**  
        - Analyze and classify only when a project has applicable strategy.
        - For each project, apply the relevant strategic action(s) based on its maturity level and provided data, following the rules and context above. Do not provide classifications or details if there is no strategy present.
        
        Format teks sebagai berikut:
        - LATAR BELAKANG: Berisi ringkasan informasi yang anda terima. Pastikan seluruh informasi penting terangkum dalam latar belakang. Abaikan informasi yang bersifat redundan.
        - ANALISA: Berisi hasil analisis yang telah anda lakukan.
        - REKOMENDASI: Berikan rekomendasi atas hasil analisa yang anda lakukan. Rekomendasi harus memenuhi prinsip SMART (Specific, Measurable, Attainable, Relevant, Time bound). Jika data tersedia, hitung berapa besar potensi peningkatan produksi yang dapat diperoleh.
        - Abaikan strategi yang tidak dapat diterapkan karena tidak ada data yang tersedia.
        - Jangan bersikap seperti asisten AI yang selalu memberikan kalimat penutup di akhir.
        """
    
    sys_prompt_3 = """
        Untuk dapat meningkatkan produksi nasional, strategi yang disusun terbagi menjadi tiga, yaitu strategi jangka pendek, jangka menengah, dan jangka panjang. Strategi jangka pendek adalah strategi yang diharapkan dapat meningkatkan produksi dalam waktu satu tahun atau kurang. Strategi jangka menengah adalah strategi yang diharapkan dapat meningkatkan produksi dalam waktu satu hingga tiga tahun. Strategi jangka panjang adalah strategi yang diharapkan dapat meningkatkan produksi dalam waktu lebih dari tiga tahun. Proyek dapat diklasifikasikan berdasarkan jangka waktu tersebut. Namun hanya jabarkan klasifikasi yang memiliki strategi.

Strategi yang dapat dilakukan sebagai berikut:
1. Jika lapangan memiliki project maturity level E0. On Production, maka cek apakah ada informasi terkait apakah laju produksi saat ini berada pada kapasitas maksimal secara teknis terhadap kemampuan reservoir maupun fasilitas produksi. Jika laju produksi belum berada pada kapasitas maksimal, maka selisih laju produksi terakhir dengan kemampuan produksi maksimal merupakan potensi penambahan produksi.
2. Cek apakah terdapat project dengan target tahun onstream sama dengan tahun ini. Proyek ini dapat meningkatkan produksi nasional jika dapat dipercepat dari tanggal rencana onstream. Perkiraan peningkatan produksi dalam persentase dapat dihitung dengan cara jumlah bulan tersisa hingga akhir tahun dari bulan percepatan onstream dibagi dengan jumlah bulan tersisa hingga akhir tahun dari bulan rencana onstream. Sebagai contoh, jika target onstream di bulan ke sepuluh kemudian dipercepat ke bulan ke delapan, maka persentase perkiraan peningkatan produksi sebesar $\left(\frac{12 - 8}{12 - 10} - 1\right) \cdot 100\% = 100\%$. Jika terdapat data perkiraan produksi untuk di tahun onstream dengan rencana awal, maka penambahan produksi di tahun tersebut dapat diketahui dengan mengalikan data perkiraan produksi dengan persentase peningkatan produksi.
3. Jika lapangan memiliki project maturity level E1. Production on Hold atau E4. Production Pending, maka sampaikan informasi apa yang menyebabkan project tersebut tidak dapat berproduksi. Cari solusi apa yang memungkinkan untuk menangani masalah yang ada dengan cepat sehingga bisa berproduksi kembali dalam waktu kurang dari satu tahun.
4. Jika lapangan memiliki project maturity level E2. Under Development, maka sampaikan informasi status pengembangan lapangan saat ini. Evaluasi apakah ada rencana pengembangan lapangan yang dapat dipercepat sehingga dapat mempercepat tanggal onstream dari rencana yang diberikan. Jika ada yang dapat dipercepat, maka perkiraan peningkatan produksi dapat menggunakan metode perkiraan berdasarkan percepatan bulan onstream.
5. Jika lapangan memiliki project maturity level E3. Justified for Development, maka sampaikan status evaluasi *Final Investment Decision* (FID) saat ini. Evaluasi apakah terdapat kendala secara keekonomian atas proyek. Jika terdapat kendala keekonomian, maka evaluasi insentif apa yang dapat diberikan pemerintah untuk memperbaiki kondisi keekonomian proyek. Insentif yang biasa diberikan antara lain pemberian split tambahan untuk kontraktor, percepatan depresiasi, penundaan pajak, pengurangan pajak. Selain itu lakukan analisa apakah FID dapat dievaluasi dengan lebih cepat atau tidak.
6. Jika lapangan memiliki project maturity level E5. Development Unclarified, maka evaluasi kendala apa yang menyebabkan lapangan tidak kunjung dikembangkan. Jika kendala berupa masalah keekonomian, evaluasi insentif apa yang tepat untuk dilakukan.
7. Jika lapangan memiliki project maturity level E6. Further Development atau E8. Further Development not Viable, maka evaluasi apakah proyek tersebut dapat diajukan untuk mendapatkan persetujuan Optimasi Pengembangan Lapangan (OPL) atau OPLL. Evaluasi percepatan apa yang dapat dilakukan berdasarkan informasi yang diberikan.
8. Jika lapangan memiliki project maturity level X0. Development Pending, maka evaluasi apakah terdapat kendala yang membuat proyek tidak kunjung diajukan untuk mendapatkan persetujuan *Plan of Development* (POD). Analisa apakah terdapat kendala keekonomian atau kendala teknikal.
9. Jika lapangan memiliki project maturity level X1. Discovery under Evaluation, maka evaluasi apakah project tersebut memerlukan tambahan sumur delineasi untuk mengetahui batas sebaran hidrokarbon. Jika *resources* lebih kecil dari 1000 MSTB atau 5 BSCF, evaluasi apakah proyek tersebut dapat diusulkan untuk memperoleh persetujuan *Put on Production* (POP).
10. Periksa apakah di lapangan tersebut terdapat proyek yang memiliki potensi *Non Producing Zone*, atau lapisan yang saat ini belum diperforasi namun memiliki potensi untuk diproduksikan.
11. Periksa apakah di lapangan tersebut terdapat proyek yang memiliki potensi untuk dapat menerapkan pemboran horizontal untuk meningkatkan produksi.

Informasi tambahan sebagai konteks:
- Data pelaporan resmi mengambil tanggal 31 Desember Pukul 23.59. Sebagai contoh, data tahun 2024 berarti merujuk pada data dengan status 31 Desember 2024 Pukul 23.59. Status ini disebut Waktu Acuan Pelaporan (WAP).
- E0. On Production adalah proyek yang pernah atau sedang berproduksi secara komersial. Sebagai contoh, Jika proyek pada pelaporan status 31 Desember 2024 memiliki project maturity level E0, maka setidaknya di tahun 2024 pernah berproduksi secara komersial walau hanya satu hari.
- E1. Production on Hold adalah proyek yang pada tahun terakhir tidak berproduksi secara komersial. Sebagai contoh, jika proyek pada pelaporan status 31 Desember 2024 memiliki project maturity level E1, maka sepanjang tahun 2024 proyek tidak berproduksi sama sekali secara komersial. Dengan kata lain, proyek tidak memiliki *lifting*. Namun proyek mungkin saja berproduksi secara teknis, misalnya dalam konteks melakukan uji produksi.
- E2. Under Development adalah proyek telah memiliki *Final Investment Decision* (FID). Biasanya pada fase ini proyek masuk dalam tahapan pembangunan fasilitas produksi.
- E3. Justified for Development adalah proyek telah memiliki persetujuan pengembangan lapangan namun belum memiliki FID. Jenis-jenis persetujuan pengembangan lapangan dapat berupa:
	- POD I untuk POD pertama di Wilayah Kerja (*Working Area*) tersebut. Persetujuan POD I diberikan oleh Menteri Energi dan Sumber Daya Mineral (ESDM).
	- POD untuk POD kedua dan selanjutnya. Persetujuan POD diberikan oleh Kepala SKK Migas.
	- Optimasi Pengembangan Lapangan (OPL). Persetujuan OPL diberikan pada proyek yang merupakan pengembangan lanjutan di lapangan tersebut. Karena OPL merupakan pengembangan lanjutan, maka biasanya lapangan tersebut sebelumnya telah memiliki proyek yang telah mendapatkan persetujuan POD atau sebelumnya pernah berproduksi secara komersial.
	- Optimasi Pengembangan Lapangan-Lapangan (OPLL). Pada prinsipnya serupa dengan persetujuan OPL. Namun perbedaannya, persetujuan OPLL diberikan untuk pengembangan yang jumlah lapangannya lebih dari satu. Karena lebih dari satu lapangan, maka tiap lapangan memiliki proyek yang merepresentasikan lingkup pekerjaan dalam OPLL tersebut.
	- *Put on Production* (POP). Persetujuan ini hanya mengizinkan proyek berproduksi dengan maksimal dua sumur produksi dengan tanpa pembangunan fasilitas produksi dan mengandalkan fasilitas produksi yang telah ada. Tujuan dari POP adalah untuk mempercepat produksi walau kegiatan eksplorasi belum selesai.
- E4. Production Pending adalah proyek yang memiliki kendala baik teknis maupun keekonomian yang cukup berat sehingga tidak dapat berproduksi secara komersial, namun masih memiliki rencana untuk kembali berproduksi di kemudian hari. Salah satu kriteria kendala ini adalah proyek setidaknya pada tiga tahun terakhir tidak berproduksi secara komersial. Sebagai contoh, jika proyek telah tidak berproduksi secara komersial sepanjang tahun 2022 hingga tahun 2024, maka pada pelaporan status 31 Desember 2024 memiliki project maturity level E4. Production Pending. Pada beberapa kasus, berdasarkan *expert judgement* bisa saja proyek langsung diberikan project maturity level E4. Production Pending walaupun kurang dari tiga tahun apabila kendala yang dihadapi dinilai cukup berat untuk diatasi.
- E5. Development Unclarified adalah proyek telah mendapatkan persetujuan pengembangan lapangan namun tidak kunjung mendapatkan FID. Atau bisa juga proyek yang telah mendapatkan FID namun kemudian FID tersebut dicabut oleh manajemen perusahaan. Secara umum, jika tidak ditetapkan lain pada dokumen persetujuan pengembangan lapangan, maka proyek yang telah melewati tiga tahun dari persetujuan pengembangan lapangan masuk dalam project maturity level E5. Sebagai contoh, jika proyek mendapatkan persetujuan POD pada tahun 2024, maka pada status pelaporan 31 Desember 2027 project maturity level menjadi E5. Hal ini karena pada status pelaporan 31 Desember 2024 hingga 31 Desember 2026 proyek telah tiga kali mendapatkan project maturity level E3. Justified for Development. Pada beberapa kasus, berdasarkan *expert judgement* bisa saja proyek langsung diberikan project maturity level E5. Development Unclarified walaupun kurang dari tiga kali pelaporan apabila kendala yang dihadapi dinilai cukup berat untuk diatasi.
- E6. Further Development adalah ide atau konsep proyek pengembangan lanjut yang belum mendapatkan persetujuan OPL atau OPLL. Konsep proyek ini merupakan potensi yang dapat menambah cadangan migas dan memberikan tambahan produksi. Proyek dengan project maturity level E6 merupakan proyek yang baru diidentifikasi sehingga masih terdapat optimisme bahwa proyek dapat diusulkan untuk dapat dikembangkan.
- E8. Further Development adalah ide atau konsep proyek pengembangan lanjut yang lebih dari satu tahun tidak kunjung diajukan untuk mendapatkan persetujuan OPL atau OPLL. Konsep proyek pengembangan lanjut jangka panjang juga dapat dicatat dengan project maturity level ini. Biasanya, walaupun tidak selalu, proyek untuk jangka panjang ini ditandai dengan kata "Future" dalam nama proyeknya. Dalam konteks ini, proyek tersebut biasanya hanya perkiraan kasar berdasarkan perkiraan *recovery factor* yang ideal yang seharusnya dapat dicapai di lapangan tersebut.
- X0. Development Pending adalah proyek yang telah memperoleh persetujuan Penentuan Status Eksplorasi (PSE). Persetujuan PSE menandakan bahwa proyek tidak lagi berada dalam fase eksplorasi karena data yang ada dianggap cukup untuk menjadi dasar dalam evaluasi pengembangan lapangan.
- X1. Discovery under Evaluation adalah proyek yang berhasil membuktikan bahwa terdapat hidrokarbon yang dapat mengalir ke permukaan secara terus-menerus (sustainable) berdasarkan tekanan alami reservoir, bantuan teknologi produksi seperti pompa, atau aktivitas tertentu. Pembuktian ini didasarkan pada data hasil uji produksi atau analisa log sumur. Proyek hanya boleh memiliki project maturity level ini selama dua kali pelaporan. Sebagai contoh, proyek yang telah memiliki project maturity level X1 pada 31 Desember 2024 dan 31 Desember 2025 maka pada 31 Desember 2026 tidak boleh lagi memiliki project maturity level X1.
- X2. Development Undetermined adalah proyek yang sebelumnya pernah memiliki project maturity level X1. Discovery under Evaluation namun setelah lebih dari dua kali pelaporan masih memerlukan data tambahan. Misalnya data tambahan yang diperlukan adalah sumur delineasi, uji produksi, dan sebagainya yang menunjang evaluasi besar sumber daya kontinjen. Selain itu, bisa juga proyek mengalami kendala keekonomian namun masih bisa dibantu melalui skema insentif.
- X3. Development not Viable adalah proyek yang sebelumnya pernah memiliki project maturity level X1. Discovery under Evaluation namun setelah dievaluasi lebih lanjut tidak dapat dikembangkan karena terkendala masalah keekonomian yang berat sehingga tidak ada skema insentif yang dapat diberikan yang mampu mengatasi masalah keekonomian tersebut.
    """

    df = run_query(table=TableName.PROJECT_RESOURCES, output=4)
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

    load_dotenv(find_dotenv())
    client = OpenAI(base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt_3},
            {"role": "user", 
             "content": f"Berikut adalah data yang perlu dianalisa: {paragraph}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


