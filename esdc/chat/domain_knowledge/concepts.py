"""Domain concepts definitions for ESDC chat agent."""

DOMAIN_CONCEPTS: dict[str, dict] = {
    "uncertainty_levels": {
        "1P": {
            "db_value": "1. Low Value",
            "description": "Proven reserves - P90 confidence",
        },
        "1R": {
            "db_value": "1. Low Value",
            "description": (
                "Low estimate GRR - P90 confidence."
                " See KSMI: ProjectClassification.Reserves"
            ),
        },
        "1C": {
            "db_value": "1. Low Value",
            "description": "Low estimate Contingent Resources",
        },
        "1U": {
            "db_value": "1. Low Value",
            "description": "Low estimate Prospective Resources",
        },
        "2P": {
            "db_value": "2. Middle Value",
            "description": "Proven + Probable reserves - P50 confidence",
        },
        "2R": {
            "db_value": "2. Middle Value",
            "description": (
                "Best estimate GRR - P50 confidence."
                " See KSMI: ProjectClassification.Reserves"
            ),
        },
        "2C": {
            "db_value": "2. Middle Value",
            "description": "Best estimate Contingent Resources",
        },
        "2U": {
            "db_value": "2. Middle Value",
            "description": "Best estimate Prospective Resources",
        },
        "3P": {
            "db_value": "3. High Value",
            "description": "Proven plus Probable plus Possible reserves - P10 confidence",  # noqa: E501
        },
        "3R": {
            "db_value": "3. High Value",
            "description": (
                "High estimate GRR - P10 confidence."
                " See KSMI: ProjectClassification.Reserves"
            ),
        },
        "3C": {
            "db_value": "3. High Value",
            "description": "High estimate Contingent Resources",
        },
        "3U": {
            "db_value": "3. High Value",
            "description": "High estimate Prospective Resources",
        },
        "proven": {"db_value": "1. Low Value", "description": "Proven/terbukti"},
        "probable": {
            "calculation": "Middle - Low",
            "description": "Probable/mungkin - difference between middle and low",
        },
        "possible": {
            "calculation": "High - Middle",
            "description": "Possible/harapan - difference between high and middle",
        },
    },
    "project_classes": {
        "reserves": {
            "db_value": None,
            "columns": ["res_*"],
            "description": "Commercial reserves only",
        },
        "grr": {
            "db_value": "1. Reserves & GRR",
            "columns": ["rec_*"],
            "description": (
                "GRR — total recoverable resources limited by"
                " technical factors only. See KSMI: Reserves"
            ),
        },
        "contingent": {
            "db_value": "2. Contingent Resources",
            "columns": ["rec_*"],
            "description": (
                "Discovered but not commercial — blocked by"
                " commercial constraints. See KSMI: ContingentResources"
            ),
        },
        "prospective": {
            "db_value": "3. Prospective Resources",
            "columns": ["rec_*", "rec_*_risked"],
            "description": (
                "Undiscovered potential — volume x GCF. See KSMI: ProspectiveResources"
            ),
        },
        "sales potential": {
            "db_value": "1. Reserves & GRR",
            "columns": ["rec_*", "res_*"],
            "calculation": "rec_* - res_*",
            "description": (
                "GRR minus Reserves — potential that could become reserves"
                " if commercial constraints resolved."
                " See KSMI: SalesPotentialResources"
            ),
        },
    },
    "forecast_types": {
        "tpf": {
            "full_name": "Total Potential Forecast",
            "columns": ["tpf_*"],
            "description": (
                "Total potential forecast volume. Sum equals rec_* (resources)."
            ),
            "equivalent_to": "resources",
        },
        "total_potential_forecast": {
            "full_name": "Total Potential Forecast",
            "columns": ["tpf_*"],
            "description": (
                "Total potential forecast volume. Sum equals rec_* (resources)."
            ),
            "equivalent_to": "resources",
        },
        "slf": {
            "full_name": "Sales Forecast",
            "columns": ["slf_*"],
            "description": "Sales forecast volume. Sum equals res_* (reserves).",
            "equivalent_to": "reserves",
        },
        "sales_forecast": {
            "full_name": "Sales Forecast",
            "columns": ["slf_*"],
            "description": "Sales forecast volume. Sum equals res_* (reserves).",
            "equivalent_to": "reserves",
        },
        "spf": {
            "full_name": "Sales Potential Forecast",
            "columns": ["spf_*"],
            "description": (
                "Potential that could become sales if commercial"
                " constraints resolved. tpf - slf."
            ),
            "calculation": "tpf_* - slf_*",
            "equivalent_to": "sales_potential",
        },
        "sales_potential_forecast": {
            "full_name": "Sales Potential Forecast",
            "columns": ["spf_*"],
            "description": (
                "Potential that could become sales if commercial"
                " constraints resolved. tpf - slf."
            ),
            "calculation": "tpf_* - slf_*",
            "equivalent_to": "sales_potential",
        },
        "crf": {
            "full_name": "Contingent Resources Forecast",
            "columns": ["crf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Contingent Resources.",  # noqa: E501
            "applies_to": "contingent_resources",
        },
        "contingent_resources_forecast": {
            "full_name": "Contingent Resources Forecast",
            "columns": ["crf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Contingent Resources.",  # noqa: E501
            "applies_to": "contingent_resources",
        },
        "prf": {
            "full_name": "Prospective Resources Forecast",
            "columns": ["prf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Prospective Resources.",  # noqa: E501
            "applies_to": "prospective_resources",
        },
        "prospective_resources_forecast": {
            "full_name": "Prospective Resources Forecast",
            "columns": ["prf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Prospective Resources.",  # noqa: E501
            "applies_to": "prospective_resources",
        },
        "ciof": {
            "full_name": "Consumed in Operation Forecast",
            "columns": ["ciof_*"],
            "description": "Profil produksi yang digunakan oleh kegiatan operasi (Fuel, Flare, Shrinkage).",  # noqa: E501
        },
        "consumed_in_operation_forecast": {
            "full_name": "Consumed in Operation Forecast",
            "columns": ["ciof_*"],
            "description": "Profil produksi yang digunakan oleh kegiatan operasi (Fuel, Flare, Shrinkage).",  # noqa: E501
        },
        "lossf": {
            "full_name": "Loss Production Forecast",
            "columns": ["lossf_*"],
            "description": "Profil loss production yang terjadi.",
        },
        "loss_production_forecast": {
            "full_name": "Loss Production Forecast",
            "columns": ["lossf_*"],
            "description": "Profil loss production yang terjadi.",
        },
    },
    "volume_types": {
        "cadangan": {
            "columns": ["res_oc", "res_an"],
            "description": "Reserves - commercial volumes",
        },
        "sumber_daya": {
            "columns": ["rec_oc", "rec_an"],
            "description": "Resources - all recoverable volumes",
        },
        "inplace": {
            "columns": ["prj_ioip", "prj_igip"],
            "description": "Initial volumes in place in the project.",
        },
        "eur": {
            "columns": ["eur_res_*", "eur_rec_*"],
            "description": "Estimated Ultimate Recovery",
        },
    },
    "substances": {
        "oil": {"columns": ["*_oil"], "description": "Crude oil"},
        "condensate": {"columns": ["*_con"], "description": "Condensate"},
        "oil_condensate": {
            "columns": ["*_oc"],
            "description": "Oil + Condensate combined",
        },
        "associated_gas": {
            "columns": ["*_ga"],
            "description": "Associated gas",
        },
        "non_associated_gas": {
            "columns": ["*_gn"],
            "description": "Non-associated gas",
        },
        "total_gas": {
            "columns": ["*_an"],
            "description": "Associated + Non-associated gas combined",
        },
    },
    "report_terms": {
        "wap": {
            "full_name": "Waktu Acuan Pelaporan",
            "description": (
                "Annual reference date (31 Dec 23:59) for project evaluation."
                " See KSMI: WaktuAcuanPelaporan"
            ),
        },
    },
    "document_types": {
        "POD": {
            "full_name": "Plan of Development",
            "description": (
                "Izin berproduksi — regulatory approval for commercial production."
                " Includes POD, POFD, OPL, OPLL, POP variants."
                " See KSMI: ProducingLicense"
            ),
        },
        "POFD": {
            "full_name": "Plan of Further Development",
            "description": (
                "POD variant for ongoing development phases. See KSMI: ProducingLicense"
            ),
        },
        "OPL": {
            "full_name": "Optimasi Pengembangan Lapangan",
            "description": (
                "POD variant for field development optimization."
                " See KSMI: ProducingLicense"
            ),
        },
        "OPLL": {
            "full_name": "Optimasi Pengembangan Lapangan - Lapangan",
            "description": (
                "POD variant for multi-field development optimization."
                " See KSMI: ProducingLicense"
            ),
        },
        "POP": {
            "full_name": "Put on Production",
            "description": (
                "POD variant for projects transitioning to production."
                " See KSMI: ProducingLicense"
            ),
        },
        "POD_I": {
            "full_name": "Plan of Development I",
            "description": (
                "First POD approved for a working area. See KSMI: ProducingLicense"
            ),
        },
        "PSE": {
            "full_name": "Penentuan Status Eksplorasi",
            "description": (
                "Dokumen penutup eksplorasi; BUKAN izin berproduksi."
                " Required for X0 transition."
                " See KSMI: DokumenPenentuanStatusEksplorasi"
            ),
        },
        "GROOVY": {
            "full_name": "Long Term Exploration and Development Strategy",
            "description": (
                "Rencana kerja jangka panjang — grants WAP dispensation."
                " See KSMI: GROOVY"
            ),
        },
    },
    "commercial_terms": {
        "TBS": {
            "full_name": "Trustee Borrowing Scheme",
            "description": (
                "Project financing scheme: pinjaman melalui trustee, dibayar"
                " kembali dari hasil penjualan produksi via rekening trustee."
                " Bukan izin berproduksi. Umumnya muncul di bagian pendanaan"
                " dokumen POD/POFD. See KSMI: TrusteeBorrowingScheme"
            ),
        },
    },
}
