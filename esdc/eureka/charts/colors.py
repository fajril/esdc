"""Oil & Gas industry color palette constants.

Follows Shell Green Oil Convention and SPE PRMS standards.
"""

# ── Petroleum Fluid Colors (Terminal-inspired) ─────────────────

FLUID_OC = "#3FB950"  # bright terminal green
FLUID_AN = "#F85149"  # bright terminal red
FLUID_OIL = "#56D364"  # lighter green
FLUID_GA = "#FF7B72"  # lighter red
FLUID_CON = "#58A6FF"  # bright blue
FLUID_NGL = "#D29922"  # amber gold

FLUID_PALETTE: dict[str, str] = {
    "Oil+Condensate": FLUID_OC,
    "Gas+NGL": FLUID_AN,
    "Oil": FLUID_OIL,
    "Gas": FLUID_GA,
    "Condensate": FLUID_CON,
    "NGL": FLUID_NGL,
}

# ── Resource Classification Colors ──────────────────────────

RESERVES_GREEN = "#3FB950"
CONTINGENT_EXPLOIT_GRAY = "#6E7681"
CONTINGENT_EXPLORE_LIGHT_GRAY = "#8B949E"
PROSPECTIVE_BLUE = "#58A6FF"

CLASS_PALETTE: dict[str, str] = {
    "Reserves": RESERVES_GREEN,
    "GRR": "#36C5F0",
    "Contingent Resources (Exploitation)": CONTINGENT_EXPLOIT_GRAY,
    "Contingent Resources (Exploration)": CONTINGENT_EXPLORE_LIGHT_GRAY,
    "Prospective Resources": PROSPECTIVE_BLUE,
}

TIMESERIES_PALETTE_OC: dict[str, str] = {
    "Reserves": "#2EA043",
    "Sales Potential Resources": "#3FB950",
    "Contingent Resources (Exploitation)": "#7EE787",
    "Contingent Resources (Exploration)": "#A5D6A7",
    "Prospective Resources": "#C5E1A5",
}

TIMESERIES_PALETTE_AN: dict[str, str] = {
    "Reserves": "#DA3633",
    "Sales Potential Resources": "#F85149",
    "Contingent Resources (Exploitation)": "#FF7B72",
    "Contingent Resources (Exploration)": "#FFA198",
    "Prospective Resources": "#FFC1B6",
}

ONSTREAM_PALETTE: dict[str, str] = {
    "Reserves": RESERVES_GREEN,
    "Contingent Resources (Exploitation)": CONTINGENT_EXPLOIT_GRAY,
    "Contingent Resources (Exploration)": CONTINGENT_EXPLORE_LIGHT_GRAY,
    "Prospective Resources": PROSPECTIVE_BLUE,
}

# ── UI Chrome (Dark Theme) ───────────────────────────────────

BG_PRIMARY = "#0A0C10"
BG_SURFACE = "#161B22"
BG_ELEVATED = "#1C2128"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED = "#7D8590"
TEXT_LABEL = "#8B949E"
BORDER_COLOR = "#21262D"
SUCCESS = "#3FB950"
WARNING = "#D29922"
DANGER = "#F85149"
INFO = "#58A6FF"

# ── Chart Template (Dark) ────────────────────────────────────

PLOTLY_LAYOUT_DEFAULTS: dict = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "JetBrains Mono, monospace",
        "color": TEXT_PRIMARY,
        "size": 12,
    },
    "margin": {"l": 50, "r": 30, "t": 30, "b": 50},
    "hoverlabel": {
        "bgcolor": BG_ELEVATED,
        "bordercolor": BORDER_COLOR,
        "font": {
            "color": TEXT_PRIMARY,
            "size": 11,
            "family": "JetBrains Mono, monospace",
        },
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "font": {"color": TEXT_PRIMARY, "size": 11},
    },
    "xaxis": {
        "gridcolor": "rgba(48,54,61,0.3)",
        "zerolinecolor": "rgba(48,54,61,0.5)",
        "tickfont": {"color": TEXT_MUTED, "size": 11},
        "showspikes": False,
    },
    "yaxis": {
        "gridcolor": "rgba(48,54,61,0.3)",
        "zerolinecolor": "rgba(48,54,61,0.5)",
        "tickfont": {"color": TEXT_MUTED, "size": 11},
        "showspikes": False,
    },
}

PLOTLY_CONFIG: dict = {
    "responsive": True,
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "sendDataToCloud",
    ],
    "displaylogo": False,
}
