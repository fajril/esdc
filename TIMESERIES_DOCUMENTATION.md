# Dokumentasi Tabel project_timeseries

Template ini berisi daftar kolom dan keyword yang perlu dijelaskan untuk integrasi dengan domain_knowledge.py

---

## 1. Struktur Tabel project_timeseries

Tabel ini berisi data **forecast produksi ke depan** (bukan data historis). Setiap baris merepresentasikan satu project pada satu tahun tertentu.

untuk minyak dan kondensat satuan: Ribu Stock Tank Barrel (MSTB)
untuk associated gas dan non associated gas: Billion Standard Cubic Feet (BSCF)

### Kolom Identitas & Lokasi

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| id | INTEGER | Primary key | 
| report_date | TEXT | Tanggal laporan operator ke sistem eSDC |
| report_year | INTEGER | Tahun laporan ke sistem eSDC. merujuk ke 31.12.XXXX | 
| is_offshore | INTEGER | Apakah offshore? (0/1) |
| basin86_id | TEXT | ID Basin (format 86) |
| basin128 | TEXT | Nama Basin (format 128) |
| province | TEXT | Nama provinsi |
| operator_group | TEXT | Grup operator |
| report_status | TEXT | Status laporan |
| wk_id | TEXT | ID Wilayah Kerja |
| wk_name | TEXT | Nama Wilayah Kerja |
| wk_area | REAL | Luas wilayah kerja |
| wk_regionisasi_ngi | TEXT | Regionisasi NGI |
| wk_area_perwakilan_skkmigas | TEXT | Area perwakilan SKK Migas |
| psc_eff_start | TEXT | Tanggal efektif PSC mulai |
| psc_eff_end | TEXT | Tanggal efektif PSC berakhir |
| wk_lat | TEXT | Latitude WK |
| wk_long | TEXT | Longitude WK |
| field_lat | TEXT | Latitude Field |
| field_long | TEXT | Longitude Field |
| wk_subgroup | TEXT | Subgroup WK |
| operator_name | TEXT | Nama operator |
| field_id | TEXT | ID Field |
| field_name | TEXT | Nama Field |
| project_id | TEXT | ID Project |
| project_name | TEXT | Nama Project |

### Kolom Status & Klasifikasi

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| project_isactive | INTEGER | Apakah project aktif? (0/1) |
| project_remarks | TEXT | Catatan project. dapat berupa insight atas proyek tersebut. |
| vol_remarks | TEXT | Catatan volume |
| frcast_remarks | TEXT | Catatan forecast |
| pod_letter_num | TEXT | Nomor surat POD |
| pod_name | TEXT | Nama POD |
| onstream_year | INTEGER | Tahun onstream |
| onstream_actual | TEXT | Tanggal onstream actual |
| project_class | TEXT | Klasifikasi project |
| project_level | TEXT | Level project |
| prod_stage | TEXT | Tahapan produksi |
| year | INTEGER | Tahun forecast |
| hist_year | INTEGER | Tahun historis |

### Kolom Cumulative Production (Historis)

Kolom Cumulative Production artinya produksi yang berasal dari pertama kali onstream ke tahun dalam report_year

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| cprd_grs_oil | REAL | Kumulatif gross production oil |
| cprd_grs_con | REAL | Kumulatif gross production condensate |
| cprd_grs_ga | REAL | Kumulatif gross production associated gas |
| cprd_grs_gn | REAL | Kumulatif gross production non-associated gas |
| cprd_grs_oc | REAL | Kumulatif gross production oil+condensate |
| cprd_grs_an | REAL | Kumulatif gross production total gas |
| cprd_sls_oil | REAL | Kumulatif sales oil |
| cprd_sls_con | REAL | Kumulatif sales condensate |
| cprd_sls_ga | REAL | Kumulatif sales associated gas |
| cprd_sls_gn | REAL | Kumulatif sales non-associated gas |
| cprd_sls_oc | REAL | Kumulatif sales oil+condensate |
| cprd_sls_an | REAL | Kumulatif sales total gas |

### Kolom Production Rate

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| rate_oil | REAL | Rate produksi oil |
| rate_con | REAL | Rate produksi condensate |
| rate_ga | REAL | Rate produksi associated gas |
| rate_gn | REAL | Rate produksi non-associated gas |
| rate_oc | REAL | Rate produksi oil+condensate |
| rate_an | REAL | Rate produksi total gas |

### Kolom Forecast - TPF (Total Potential Forecast)

**Singkatan: TPF = Total Potential Forecast.** Kolom ini merupakan profil perkiraan produksi. jika kita jumlahkan, maka jumlah volume seluruh TPF di seluruh tahun adalah sama besar dengan kolom rec_*.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| tpf_oil | REAL | Total Potential Forecast oil |
| tpf_con | REAL | Total Potential Forecast condensate |
| tpf_ga | REAL | Total Potential Forecast associated gas |
| tpf_gn | REAL | Total Potential Forecast non-associated gas |
| tpf_oc | REAL | Total Potential Forecast oil+condensate |
| tpf_an | REAL | Total Potential Forecast total gas |
| tpf_risked_oil | REAL | Forecast oil (risked) |
| tpf_risked_con | REAL | Forecast condensate (risked) |
| tpf_risked_ga | REAL | Forecast associated gas (risked) |
| tpf_risked_gn | REAL | Forecast non-associated gas (risked) |
| tpf_risked_oc | REAL | Forecast oil+condensate (risked) |
| tpf_risked_an | REAL | Forecast total gas (risked) |

### Kolom Forecast - SLF (Sales Forecast)

**Singkatan: SLF = Sales Forecast.** Kolom ini merupakan profil perkiraan produksi reserves. jika kita jumlahkan, maka jumlah volume seluruh SLF di seluruh tahun adalah sama besar dengan kolom res_*.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| slf_oil | REAL | Sales Forecast oil |
| slf_con | REAL | Sales Forecast condensate |
| slf_ga | REAL | Sales Forecast associated gas |
| slf_gn | REAL | Sales Forecast non-associated gas |
| slf_oc | REAL | Sales Forecast oil+condensate |
| slf_an | REAL | Sales Forecast total gas |

### Kolom Forecast - SPF (Sales Potential Forecast)

**Singkatan: SPF = Sales Potential Forecast.** Kolom ini merupakan selisih antara TPF dan SLF. artinya, profil SPF merupakan potensi yang bisa diproduksikan andaikata kendala komersial dapat diatasi.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| spf_oil | REAL | Sales Potential Forecast oil |
| spf_con | REAL | Sales Potential Forecast condensate |
| spf_ga | REAL | Sales Potential Forecast associated gas |
| spf_gn | REAL | Sales Potential Forecast non-associated gas |
| spf_oc | REAL | Sales Potential Forecast oil+condensate |
| spf_an | REAL | Sales Potential Forecast total gas |

### Kolom Forecast - CRF (Contingent Resources Forecast)

**Singkatan: CRF = Contingent Resources Forecast.** Kolom ini merupakan perkiraan profil produksi untuk proyek dengan klasifikasi Contingent Resources. penjumlahan kolom ini untuk proyek yang sama akan menghasilkan volume yang sama dengan kolom rec_* apabila proyek memiliki klasifikasi Contingent Resources.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| crf_oil | REAL | Contingent Resources Forecast oil |
| crf_con | REAL | Contingent Resources Forecast condensate |
| crf_ga | REAL | Contingent Resources Forecast associated gas |
| crf_gn | REAL | Contingent Resources Forecast non-associated gas |
| crf_oc | REAL | Contingent Resources Forecast oil+condensate |
| crf_an | REAL | Contingent Resources Forecast total gas |

### Kolom Forecast - PRF (Prospective Resources Forecast)

**Singkatan: PRF = Prospective Resources Forecast.** Kolom ini merupakan perkiraan profil produksi untuk proyek dengan klasifikasi Prospective Resources. penjumlahan kolom ini untuk proyek yang sama akan menghasilkan volume yang sama dengan kolom rec_* apabila proyek memiliki klasifikasi Prospective Resources.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| prf_oil | REAL | Prospective Resources Forecast oil |
| prf_con | REAL | Prospective Resources Forecast condensate |
| prf_ga | REAL | Prospective Resources Forecast associated gas |
| prf_gn | REAL | Prospective Resources Forecast non-associated gas |
| prf_oc | REAL | Prospective Resources Forecast oil+condensate |
| prf_an | REAL | Prospective Resources Forecast total gas |

### Kolom Forecast - CIOF (Consumed in Operation Forecast)

**Singkatan: CIOF = Consumed in Operation Forecast.** Kolom ini merupakan perkiraan profil produksi yang digunakan oleh kegiatan operasi. Komponen CIO adalah Fuel, Flare, Shrinkage.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| ciof_oil | REAL | Consumed in Operation Forecast oil |
| ciof_con | REAL | Consumed in Operation Forecast condensate |
| ciof_ga | REAL | Consumed in Operation Forecast associated gas |
| ciof_gn | REAL | Consumed in Operation Forecast non-associated gas |
| ciof_oc | REAL | Consumed in Operation Forecast oil+condensate |
| ciof_an | REAL | Consumed in Operation Forecast total gas |

### Kolom Forecast - LOSSF (Loss Production Forecast)

**Singkatan: LOSSF = Loss Production Forecast.** Kolom ini merupakan perkiraan profil dari loss production yang terjadi.

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| lossf_oil | REAL | Loss Production Forecast oil |
| lossf_con | REAL | Loss Production Forecast condensate |
| lossf_ga | REAL | Loss Production Forecast associated gas |
| lossf_gn | REAL | Loss Production Forecast non-associated gas |
| lossf_oc | REAL | Loss Production Forecast oil+condensate |
| lossf_an | REAL | Loss Production Forecast total gas |

### Kolom Lainnya

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| ghv | REAL | Gas Heating Value |

---

## 2. Synonyms & Keywords untuk Domain Knowledge

### Bahasa Indonesia → Mapping

| Bahasa Indonesia | Bahasa Inggris | Mapping ke Kolom/Query |
|------------------|--------------|----------------------|
| perkiraan | forecast | tpf_* |
| proyeksi | projection | perkiraan |
| produksi kedepan | future production | forecast |
| puncak produksi | peak production | MAX(tpf_*) |
| mendatar (plateau) | plateau | produksi yang angkanya sama, hampir tidak berubah naik atau turun |
| penurunan | decline | fase produksi yang angkanya menurun |
| masa akhir produksi | end of life | tahun terakhir tpf_* > 0 |
| End of Life | EOL | tahun terakhir tpf_* > 0 |
| masa akhir produksi | last production year | MAX(year) WHERE tpf_oil > 0 |
| tahun onstream | onstream | onstream_year |
| tahun mulai produksi | start production | onstream_year |
| peningkatan | ramp up | fase produksi yang angkanya meningkat setelah onstream |
| penutupan sumur | shut in | well yang sementara tidak berproduksi |

### Substance/Fluids

| Bahasa Indonesia | Bahasa Inggris | Suffix |
|------------------|--------------|--------|
| Minyak | Oil | _oil |
| Kondensat | Condensate | _con |
| Minyak + Kondensat | Oil + Condensate | _oc |
| Gas berasosiasi | Associated Gas | _ga |
| Gas non-berasosiasi | Non-associated Gas | _gn |
| Gas total | Total Gas | _an |

---

## 3. Use Case Query yang Perlu Dihandle

### Use Case 1: Peak Production
**Pertanyaan:** "Kapan peak production Duri?"

**SQL Pattern:**
```sql
SELECT 
    year,
    tpf_oc,
    tpf_an
FROM project_timeseries
WHERE project_name LIKE '%Duri%'
    AND tpf_oc = (
        SELECT MAX(tpf_oc) 
        FROM project_timeseries 
        WHERE project_name LIKE '%Duri%'
    );
```

**Catatan:** Mencari tahun dengan nilai TPF tertinggi untuk suatu project.

### Use Case 2: Last Production Year
**Pertanyaan:** "Sampai tahun berapa Duri masih berproduksi?"

**SQL Pattern:**
```sql
SELECT MAX(year) as last_production_year
FROM project_timeseries
WHERE project_name LIKE '%Duri%'
    AND (tpf_oil > 0 OR tpf_an > 0);
```

**Catatan:** Mencari tahun terakhir dimana masih ada produksi (tpf > 0).

### Use Case 3: Onstream Year
**Pertanyaan:** "Kapan Duri mulai onstream?"

**SQL Pattern:**
```sql
-- Option A: Gunakan kolom onstream_year (jika tersedia)
SELECT DISTINCT 
    COALESCE(onstream_year, 
        (SELECT MIN(year) 
         FROM project_timeseries AS pt2 
         WHERE pt2.project_name = pt1.project_name 
         AND (tpf_oil > 0 OR tpf_an > 0))) as onstream_year
FROM project_timeseries AS pt1
WHERE project_name LIKE '%Duri%';
```

**Catatan:** Gunakan onstream_year jika ada, fallback ke MIN(year) dengan tpf > 0.

### Use Case 4: Forecast Volume per Tahun
**Pertanyaan:** "Berapa forecast produksi Duri tahun 2025?"

**SQL Pattern:**
```sql
SELECT 
    project_name,
    year,
    tpf_oc as forecast_oil_condensate_mstb,
    tpf_an as forecast_gas_bscf,
    ROUND(tpf_oc * 1000 / CASE 
        WHEN year % 4 = 0 THEN 366 
        ELSE 365 
    END, 2) as forecast_oil_condensate_bopd,
    ROUND(tpf_an * 1000 / CASE 
        WHEN year % 4 = 0 THEN 366 
        ELSE 365 
    END, 2) as forecast_gas_mmscfd
FROM project_timeseries
WHERE project_name LIKE '%Duri%' 
    AND year = 2025;
```

**Catatan:** 
- Output dalam satuan MSTB dan BSCF
- Konversi ke BOPD (Barrel Oil Per Day) dan MMSCFD (Million Standard Cubic Feet per Day)
- BOPD = MSTB * 1000 / jumlah hari dalam tahun
- MMSCFD = BSCF * 1000 / jumlah hari dalam tahun

### Use Case 5: Production Trend
**Pertanyaan:** "Bagaimana trend produksi Duri 5 tahun ke depan?"

**SQL Pattern:**
```sql
SELECT 
    year,
    tpf_oc,
    tpf_an
FROM project_timeseries
WHERE project_name LIKE '%Duri%'
    AND year >= (SELECT MAX(report_year) FROM project_timeseries)
    AND year <= (SELECT MAX(report_year) FROM project_timeseries) + 5
ORDER BY year;
```

**Catatan:** Mengambil data dari report_year terakhir dan 5 tahun ke depan.

---

## 4. View/Aggregasi yang Diperlukan

Apakah perlu membuat view seperti:

- [x] `field_timeseries` - Aggregate forecast per field per year
- [x] `wa_timeseries` - Aggregate forecast per WA per year  
- [x] `nkri_timeseries` - Aggregate forecast nasional per year

**Pertimbangan:**
- View akan mempercepat query untuk analisis aggregat
- View perlu dibuat dengan GROUP BY field_name/year, wk_name/year, atau total/year
- Untuk field_timeseries: SUM(tpf_*) GROUP BY report_year, field_name, year
- Untuk wa_timeseries: SUM(tpf_*) GROUP BY report_year, wk_name, year
- Untuk nkri_timeseries: SUM(tpf_*) GROUP BY report_year, year

### Contoh View: field_timeseries
```sql
CREATE VIEW field_timeseries AS
SELECT 
    report_year,
    field_name,
    wk_name,
    year,
    SUM(tpf_oil) as tpf_oil,
    SUM(tpf_con) as tpf_con,
    SUM(tpf_ga) as tpf_ga,
    SUM(tpf_gn) as tpf_gn,
    SUM(tpf_oc) as tpf_oc,
    SUM(tpf_an) as tpf_an,
    -- ... semua kolom forecast lainnya
    SUM(slf_oil) as slf_oil,
    -- ... dll
FROM project_timeseries
GROUP BY report_year, field_name, wk_name, year;
```

---

## 5. Perbedaan Forecast vs Historical

### Historical Data
- Kolom: `cprd_*`, `rate_*`
- Tahun: `<= hist_year`
- Data actual/faktual

### Forecast Data
- Kolom: `tpf_*`, `slf_*`, `spf_*`, `crf_*`, `prf_*`, `ciof_*`, `lossf_*`
- Tahun: `> hist_year`
- Data prediksi/proyeksi

### Pertanyaan & Jawaban:
1. **Apakah user perlu diberitahu secara eksplisit bahwa data adalah forecast?**
   - Ya. Contoh: report year 2023 memiliki forecast tahun 2024, padahal report year terkini sudah 2024.
   - Saran: selalu sertakan `report_year` dalam output untuk konteks.

2. **Bagaimana membedakan query historis vs forecast?**
   - Forecast: user menggunakan kata "forecast", "proyeksi", "perkiraan"
   - Historis: user menanyakan "produksi tahun X" dimana X <= report_year

3. **Apakah ada kolom yang menandai apakah suatu tahun adalah historis atau forecast?**
   - Kolom `year` mengandung data forecast
   - Kolom `hist_year` untuk data historis (bisa jadi sama dengan report_year)
   - Jika `year` > `hist_year` atau `year` > `report_year`, maka itu forecast

---

## 6. Column Groups untuk Domain Knowledge

### Timeseries Column Groups

```python
COLUMN_GROUPS: Dict[str, List[str]] = {
    "timeseries_forecast_tpf": [
        "tpf_oil", "tpf_con", "tpf_ga", "tpf_gn", "tpf_oc", "tpf_an"
    ],
    "timeseries_forecast_tpf_risked": [
        "tpf_risked_oil", "tpf_risked_con", "tpf_risked_ga", 
        "tpf_risked_gn", "tpf_risked_oc", "tpf_risked_an"
    ],
    "timeseries_forecast_slf": [
        "slf_oil", "slf_con", "slf_ga", "slf_gn", "slf_oc", "slf_an"
    ],
    "timeseries_forecast_spf": [
        "spf_oil", "spf_con", "spf_ga", "spf_gn", "spf_oc", "spf_an"
    ],
    "timeseries_forecast_crf": [
        "crf_oil", "crf_con", "crf_ga", "crf_gn", "crf_oc", "crf_an"
    ],
    "timeseries_forecast_prf": [
        "prf_oil", "prf_con", "prf_ga", "prf_gn", "prf_oc", "prf_an"
    ],
    "timeseries_forecast_ciof": [
        "ciof_oil", "ciof_con", "ciof_ga", "ciof_gn", "ciof_oc", "ciof_an"
    ],
    "timeseries_forecast_lossf": [
        "lossf_oil", "lossf_con", "lossf_ga", "lossf_gn", "lossf_oc", "lossf_an"
    ],
    "timeseries_historical_cumulative": [
        "cprd_grs_oil", "cprd_grs_con", "cprd_grs_ga", "cprd_grs_gn", "cprd_grs_oc", "cprd_grs_an",
        "cprd_sls_oil", "cprd_sls_con", "cprd_sls_ga", "cprd_sls_gn", "cprd_sls_oc", "cprd_sls_an"
    ],
    "timeseries_historical_rate": [
        "rate_oil", "rate_con", "rate_ga", "rate_gn", "rate_oc", "rate_an"
    ],
}
```

### Synonyms untuk Timeseries

```python
TIMESERIES_SYNONYMS: Dict[str, str] = {
    # Forecast
    "perkiraan": "forecast",
    "forecast": "forecast",
    "proyeksi": "forecast",
    "projection": "forecast",
    "produksi kedepan": "forecast",
    "future production": "forecast",
    "ramalan": "forecast",
    
    # Peak production
    "puncak produksi": "peak_production",
    "peak production": "peak_production",
    "produksi maksimum": "peak_production",
    "maximum production": "peak_production",
    
    # Plateau
    "plateau": "plateau",
    "mendatar": "plateau",
    "produksi stabil": "plateau",
    "stable production": "plateau",
    
    # Decline
    "decline": "decline",
    "penurunan": "decline",
    "produksi menurun": "decline",
    "falling production": "decline",
    
    # Ramp up
    "ramp up": "ramp_up",
    "peningkatan": "ramp_up",
    "naik produksi": "ramp_up",
    "increasing production": "ramp_up",
    
    # End of life
    "end of life": "eol",
    "eol": "eol",
    "masa akhir produksi": "eol",
    "last production year": "eol",
    
    # Onstream
    "onstream": "onstream",
    "on stream": "onstream",
    "mulai produksi": "onstream",
    "start production": "onstream",
    "tahun onstream": "onstream",
    
    # Shut in
    "shut in": "shut_in",
    "penutupan sumur": "shut_in",
    "well shut": "shut_in",
}
```

---

## 7. Catatan Implementasi

Setelah dokumen ini dilengkapi, perlu diimplementasikan di:

1. **domain_knowledge.py**
   - [x] Synonyms untuk timeseries keywords (sudah didefinisikan di atas)
   - [x] Column groups untuk timeseries columns (sudah didefinisikan di atas)
   - [x] SQL patterns untuk use cases (sudah didefinisikan di atas)
   - [ ] Function untuk peak production detection
   - [ ] Function untuk EOL detection
   - [ ] Function untuk konversi satuan (MSTB → BOPD, BSCF → MMSCFD)

2. **SQL View (opsional)**
   - [ ] Buat view `field_timeseries`
   - [ ] Buat view `wa_timeseries`
   - [ ] Buat view `nkri_timeseries`

3. **Chat Agent**
   - [ ] Update system prompt untuk mengenali timeseries queries
   - [ ] Update query builder untuk generate SQL yang tepat
   - [ ] Tambahkan handler untuk pertanyaan dengan unit conversion

---

**Status:** Dokumentasi lengkap - Siap diimplementasikan
**Last Updated:** 2025-03-30
