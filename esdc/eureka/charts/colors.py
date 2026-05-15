"""Oil & Gas industry color palette constants.

Follows Shell Green Oil Convention and SPE PRMS standards.
"""

# --- Petroleum Fluid Colors (Shell Green Oil Convention) ---

OIL_DARK_GREEN = "#2E7D32"
OIL_FOREST_GREEN = "#4CAF50"
GAS_CRIMSON_RED = "#C62828"
GAS_RED = "#EF5350"
CONDENSATE_STEEL_BLUE = "#1565C0"
NGL_AMBER_GOLD = "#D4A843"

FLUID_OC = OIL_DARK_GREEN
FLUID_AN = GAS_CRIMSON_RED
FLUID_OIL = OIL_FOREST_GREEN
FLUID_GA = GAS_RED
FLUID_CON = CONDENSATE_STEEL_BLUE
FLUID_NGL = NGL_AMBER_GOLD

FLUID_PALETTE: dict[str, str] = {
    "Oil+Condensate": FLUID_OC,
    "Gas+NGL": FLUID_AN,
    "Oil": FLUID_OIL,
    "Gas": FLUID_GA,
    "Condensate": FLUID_CON,
    "NGL": FLUID_NGL,
}

# --- Resource Classification Colors (SPE PRMS) ---

RESERVES_BLUE = "#1565C0"
CONTINGENT_EXPLOIT_AMBER = "#F9A825"
CONTINGENT_EXPLORE_LIGHT_AMBER = "#FFCA28"
PROSPECTIVE_ORANGE = "#E65100"

CLASS_PALETTE: dict[str, str] = {
    "Reserves & GRR": RESERVES_BLUE,
    "Contingent Exploitation": CONTINGENT_EXPLOIT_AMBER,
    "Contingent Exploration": CONTINGENT_EXPLORE_LIGHT_AMBER,
    "Prospective Resources": PROSPECTIVE_ORANGE,
}

# --- Project Stage Colors ---

PRODUCING_TEAL = "#00897B"
DEVELOPMENT_INDIGO = "#3949AB"
EXPLORATION_ORANGE = PROSPECTIVE_ORANGE

STAGE_PALETTE: dict[str, str] = {
    "Producing": PRODUCING_TEAL,
    "Development": DEVELOPMENT_INDIGO,
    "Exploration": EXPLORATION_ORANGE,
}

# --- UI Chrome ---

NAV_DEEP_NAVY = "#1B2A4A"
CARD_WHITE = "#FFFFFF"
PAGE_LIGHT_GRAY = "#F5F6FA"
BORDER_SILVER = "#DEE2E6"
TEXT_CHARCOAL = "#2C3E50"
TEXT_SLATE = "#7F8C8D"
SUCCESS_EMERALD = "#27AE60"
WARNING_AMBER = "#F39C12"
DANGER_RED = "#E74C3C"

# --- YoY Change Indicators ---

YOY_UP = SUCCESS_EMERALD
YOY_DOWN = DANGER_RED
YOY_FLAT = TEXT_SLATE

# --- Chart Template ---

PLOTLY_LAYOUT_DEFAULTS: dict = {
    "paper_bgcolor": CARD_WHITE,
    "plot_bgcolor": PAGE_LIGHT_GRAY,
    "font": {
        "family": ("Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        "color": TEXT_CHARCOAL,
        "size": 13,
    },
    "margin": {"l": 60, "r": 30, "t": 40, "b": 50},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
}

PLOTLY_CONFIG: dict = {
    "responsive": True,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
}
