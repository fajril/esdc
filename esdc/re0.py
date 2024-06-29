from typing import Tuple
import pandas as pd

from .validate import Severity

def re0007(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0007 - Oil GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{\text{P90}} \geq 0$$
    """
    invalid = ~(
        (project_resources["rec_oil"] >= 0)
        & (project_resources["uncert_lvl"] == "1. Low Value")
    )
    return "rec_oil", "RE0007", Severity.STRICT, invalid


def re0008(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0008 - Condensate GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{c \text{ P90}} \geq 0$$

    """
    invalid = ~(
        (project_resources["rec_con"] >= 0)
        & (project_resources["uncert_lvl"] == "1. Low Value")
    )
    return "rec_con", "RE0008", Severity.STRICT, invalid

def re0009(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0009 - Associated Gas GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{a \text{ P90}} \geq 0$$
    """
    invalid = ~(
        (project_resources["rec_ga"] >= 0)
        & (project_resources["uncert_lvl"] == "1. Low Value")
    )
    return "rec_ga", "RE0009", Severity.STRICT, invalid

def re0010(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0010 - Non Associated Gas GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{\text{P90}} \geq 0$$
    """
    invalid = ~(
        (project_resources["rec_gn"] >= 0)
        & (project_resources["uncert_lvl"] == "1. Low Value")
    )
    return "rec_gn", "RE0010", Severity.STRICT, invalid

def re0011(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0011 - Oil Reserves: 1P must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{ps}^{\text{1P}} \geq 0$$
    """
    invalid = ~(
        (project_resources["res_oil"] >= 0)
        & (project_resources["uncert_lvl"] == "1. Low Value")
    )
    return "rec_gn", "RE0011", Severity.STRICT, invalid

def re0012(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0012 - Condensate Reserves: 1P must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{ps}^{c\text{ 1P}} \geq 0$$
    """
    invalid = ~(
        (project_resources["res_con"] >= 0)
        & (project_resources["uncert_lvl"] == "1. Low Value")
    )
    return "rec_gn", "RE0012", Severity.STRICT, invalid