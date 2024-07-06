import logging
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
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "rec_oil"
            ]
            >= 0
        )
    )
    logging.debug("RE0007: %s %s", invalid.sum(), severity.value)
    return "rec_oil", "RE0007", severity, invalid


def re0008(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0008 - Condensate GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{c \text{ P90}} \geq 0$$

    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "rec_con"
            ]
            >= 0
        )
    )
    logging.debug("RE0008: %s %s", invalid.sum(), severity.value)
    return "rec_con", "RE0008", severity, invalid


def re0009(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0009 - Associated Gas GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{a \text{ P90}} \geq 0$$
    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "rec_ga"
            ]
            >= 0
        )
    )
    logging.debug("RE0009: %s %s", invalid.sum(), severity.value)
    return "rec_ga", "RE0009", severity, invalid


def re0010(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0010 - Non Associated Gas GRR/CR/PR: 1R/1C/1U must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{\text{P90}} \geq 0$$
    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "rec_gn"
            ]
            >= 0
        )
    )
    logging.debug("RE0010: %s %s", invalid.sum(), severity.value)
    return "rec_gn", "RE0010", severity, invalid


def re0011(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0011 - Oil Reserves: 1P must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{ps}^{\text{1P}} \geq 0$$
    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "res_oil"
            ]
            >= 0
        )
    )
    logging.debug("RE0011: %s %s", invalid.sum(), severity.value)
    return "res_oil", "RE0011", severity, invalid


def re0012(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0012 - Condensate Reserves: 1P must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{ps}^{c\text{ 1P}} \geq 0$$
    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "res_con"
            ]
            >= 0
        )
    )
    logging.debug("RE0012: %s %s", invalid.sum(), severity.value)
    return "res_con", "RE0012", severity, invalid
