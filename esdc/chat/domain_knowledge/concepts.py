"""Domain concepts definitions for ESDC chat agent."""

DOMAIN_CONCEPTS: dict[str, dict] = {
    "uncertainty_levels": {
        "1P": {
            "db_value": "1. Low Value",
            "description": "Proven reserves - P90 confidence",
        },
        "1R": {
            "db_value": "1. Low Value",
            "description": "Low estimate GRR - P90 confidence",
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
            "description": "Best estimate GRR - P50 confidence",
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
            "description": "Proven plus Probable plus Possible reserves - P10 confidence",
        },
        "3R": {
            "db_value": "3. High Value",
            "description": "High estimate GRR - P10 confidence",
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
            "description": "Government of Indonesia Recoverable Resources - total recoverable resources for government share, including reserves and sales potential",
        },
        "contingent": {
            "db_value": "2. Contingent Resources",
            "columns": ["rec_*"],
            "description": "Contingent Resources - discovered but not commercial",
        },
        "prospective": {
            "db_value": "3. Prospective Resources",
            "columns": ["rec_*", "rec_*_risked"],
            "description": "Prospective Resources - undiscovered potential. Risked means the resources is multiplied by total_gcf column",
        },
        "sales potential": {
            "db_value": "1. Reserves & GRR",
            "columns": ["rec_*", "res_*"],
            "calculation": "rec_* - res_*",
            "description": "Sales Potential - resources that could be produced if commercial constraints are resolved (GRR - Reserves)",
        },
    },
    "forecast_types": {
        "tpf": {
            "full_name": "Total Potential Forecast",
            "columns": ["tpf_*"],
            "description": "Profil perkiraan produksi. Jumlah seluruh TPF sama dengan kolom rec_* (Resources).",
            "equivalent_to": "resources",
        },
        "total_potential_forecast": {
            "full_name": "Total Potential Forecast",
            "columns": ["tpf_*"],
            "description": "Profil perkiraan produksi. Jumlah seluruh TPF sama dengan kolom rec_* (Resources).",
            "equivalent_to": "resources",
        },
        "slf": {
            "full_name": "Sales Forecast",
            "columns": ["slf_*"],
            "description": "Profil perkiraan produksi reserves. Jumlah seluruh SLF sama dengan kolom res_* (Reserves).",
            "equivalent_to": "reserves",
        },
        "sales_forecast": {
            "full_name": "Sales Forecast",
            "columns": ["slf_*"],
            "description": "Profil perkiraan produksi reserves. Jumlah seluruh SLF sama dengan kolom res_* (Reserves).",
            "equivalent_to": "reserves",
        },
        "spf": {
            "full_name": "Sales Potential Forecast",
            "columns": ["spf_*"],
            "description": "Selisih antara TPF dan SLF. Potensi yang bisa diproduksikan andaikata kendala komersial dapat diatasi.",
            "calculation": "tpf_* - slf_*",
            "equivalent_to": "sales_potential",
        },
        "sales_potential_forecast": {
            "full_name": "Sales Potential Forecast",
            "columns": ["spf_*"],
            "description": "Selisih antara TPF dan SLF. Potensi yang bisa diproduksikan andaikata kendala komersial dapat diatasi.",
            "calculation": "tpf_* - slf_*",
            "equivalent_to": "sales_potential",
        },
        "crf": {
            "full_name": "Contingent Resources Forecast",
            "columns": ["crf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Contingent Resources.",
            "applies_to": "contingent_resources",
        },
        "contingent_resources_forecast": {
            "full_name": "Contingent Resources Forecast",
            "columns": ["crf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Contingent Resources.",
            "applies_to": "contingent_resources",
        },
        "prf": {
            "full_name": "Prospective Resources Forecast",
            "columns": ["prf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Prospective Resources.",
            "applies_to": "prospective_resources",
        },
        "prospective_resources_forecast": {
            "full_name": "Prospective Resources Forecast",
            "columns": ["prf_*"],
            "description": "Profil perkiraan produksi untuk proyek dengan klasifikasi Prospective Resources.",
            "applies_to": "prospective_resources",
        },
        "ciof": {
            "full_name": "Consumed in Operation Forecast",
            "columns": ["ciof_*"],
            "description": "Profil produksi yang digunakan oleh kegiatan operasi (Fuel, Flare, Shrinkage).",
        },
        "consumed_in_operation_forecast": {
            "full_name": "Consumed in Operation Forecast",
            "columns": ["ciof_*"],
            "description": "Profil produksi yang digunakan oleh kegiatan operasi (Fuel, Flare, Shrinkage).",
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
}
