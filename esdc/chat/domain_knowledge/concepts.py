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
                "Low estimate GRR (Government Recoverable Resources)"
                " - P90 confidence. R = GRR suffix."
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
                "Best estimate GRR (Government Recoverable Resources)"
                " - P50 confidence. R = GRR suffix, meaning 2R is the"
                " mid-value estimate of GRR."
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
                "High estimate GRR (Government Recoverable Resources)"
                " - P10 confidence. R = GRR suffix."
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
            "description": "Government of Indonesia Recoverable Resources - total recoverable resources for government share, including reserves and sales potential",  # noqa: E501
        },
        "contingent": {
            "db_value": "2. Contingent Resources",
            "columns": ["rec_*"],
            "description": "Contingent Resources - discovered but not commercial",
        },
        "prospective": {
            "db_value": "3. Prospective Resources",
            "columns": ["rec_*", "rec_*_risked"],
            "description": "Prospective Resources - undiscovered potential. Risked means the resources is multiplied by total_gcf column",  # noqa: E501
        },
        "sales potential": {
            "db_value": "1. Reserves & GRR",
            "columns": ["rec_*", "res_*"],
            "calculation": "rec_* - res_*",
            "description": "Sales Potential - resources that could be produced if commercial constraints are resolved (GRR - Reserves)",  # noqa: E501
        },
    },
    "forecast_types": {
        "tpf": {
            "full_name": "Total Potential Forecast",
            "columns": ["tpf_*"],
            "description": "Profil perkiraan produksi. Jumlah seluruh TPF sama dengan kolom rec_* (Resources).",  # noqa: E501
            "equivalent_to": "resources",
        },
        "total_potential_forecast": {
            "full_name": "Total Potential Forecast",
            "columns": ["tpf_*"],
            "description": "Profil perkiraan produksi. Jumlah seluruh TPF sama dengan kolom rec_* (Resources).",  # noqa: E501
            "equivalent_to": "resources",
        },
        "slf": {
            "full_name": "Sales Forecast",
            "columns": ["slf_*"],
            "description": "Profil perkiraan produksi reserves. Jumlah seluruh SLF sama dengan kolom res_* (Reserves).",  # noqa: E501
            "equivalent_to": "reserves",
        },
        "sales_forecast": {
            "full_name": "Sales Forecast",
            "columns": ["slf_*"],
            "description": "Profil perkiraan produksi reserves. Jumlah seluruh SLF sama dengan kolom res_* (Reserves).",  # noqa: E501
            "equivalent_to": "reserves",
        },
        "spf": {
            "full_name": "Sales Potential Forecast",
            "columns": ["spf_*"],
            "description": "Selisih antara TPF dan SLF. Potensi yang bisa diproduksikan andaikata kendala komersial dapat diatasi.",  # noqa: E501
            "calculation": "tpf_* - slf_*",
            "equivalent_to": "sales_potential",
        },
        "sales_potential_forecast": {
            "full_name": "Sales Potential Forecast",
            "columns": ["spf_*"],
            "description": "Selisih antara TPF dan SLF. Potensi yang bisa diproduksikan andaikata kendala komersial dapat diatasi.",  # noqa: E501
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
                "Waktu Acuan Pelaporan (WAP) is the annual reference date for"
                " reporting, set at December 31, 23:59 each year. Written as"
                " 31.12.20XX or abbreviated as just the year (20XX). Data"
                " 'tahun 2024' means WAP 31.12.2024. Project status evaluations"
                " are calculated at WAP. Previously the reference was January 1,"
                " 00:00 of the following year."
            ),
        },
    },
    "document_types": {
        "POD": {
            "full_name": "Plan of Development",
            "description": (
                "Plan of Development - regulatory development plan for oil &"
                " gas projects. Approved by Minister of ESDM. The pod_name"
                " column contains the specific POD variant name."
            ),
        },
        "POFD": {
            "full_name": "Plan of Further Development",
            "description": (
                "Plan of Further Development - POD variant for ongoing"
                " development phases of existing projects."
            ),
        },
        "OPL": {
            "full_name": "Optimasi Pengembangan Lapangan",
            "description": (
                "Optimasi Pengembangan Lapangan - POD variant for field"
                " development optimization."
            ),
        },
        "OPLL": {
            "full_name": "Optimasi Pengembangan Lapangan - Lapangan",
            "description": (
                "Optimasi Pengembangan Lapangan - Lapangan - POD variant"
                " for multi-field development optimization."
            ),
        },
        "POP": {
            "full_name": "Put on Production",
            "description": (
                "Put on Production - POD variant for projects transitioning"
                " to production. Narrow scope within POD classification."
            ),
        },
        "POD_I": {
            "full_name": "Plan of Development I",
            "description": (
                "Plan of Development I - the first POD approved for a working"
                " area. Approved by Minister of ESDM."
            ),
        },
        "PSE": {
            "full_name": "Penentuan Status Eksplorasi",
            "description": (
                "Penentuan Status Eksplorasi - determination of exploration"
                " status for a project. The is_pse_approved column indicates"
                " whether a project has PSE approved."
            ),
        },
    },
}
