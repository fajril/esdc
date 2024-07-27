import logging
from typing import Tuple
import pandas as pd

from esdc.validate import Severity


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
    logging.debug("%s: %s %s", re0007.__name__, invalid.sum(), severity.value)
    return "rec_oil", re0007.__name__, severity, invalid


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
    logging.debug("%s: %s %s", re0008.__name__, invalid.sum(), severity.value)
    return "rec_con", re0008.__name__, severity, invalid


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
    logging.debug("%s: %s %s", re0009.__name__, invalid.sum(), severity.value)
    return "rec_ga", re0009.__name__, severity, invalid


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
    logging.debug("%s: %s %s", re0010.__name__, invalid.sum(), severity.value)
    return "rec_gn", re0010.__name__, severity, invalid


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
    logging.debug("%s: %s %s", re0011.__name__, invalid.sum(), severity.value)
    return "res_oil", re0011.__name__, severity, invalid


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
    logging.debug("%s: %s %s", re0012.__name__, invalid.sum(), severity.value)
    return "res_con", re0012.__name__, severity, invalid


def re0013(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0011 - Associated Gas Reserves: 1P must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{ps}^{a \text{ 1P}} \geq 0$$
    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "res_ga"
            ]
            >= 0
        )
    )
    logging.debug("%s: %s %s", re0013.__name__, invalid.sum(), severity.value)
    return "res_ga", re0013.__name__, severity, invalid


def re0014(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0014 - Non Associated Gas Reserves: 1P must be higher than or equal to 0

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{ps}^{\text{ 1P}} \geq 0$$
    """
    severity = Severity.STRICT
    invalid = ~(
        (
            project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
                "res_gn"
            ]
            >= 0
        )
    )
    logging.debug("%s: %s %s", re0014.__name__, invalid.sum(), severity.value)
    return "res_gn", re0014.__name__, severity, invalid


def re0015(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0015 - Oil GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{\text{P90}} \leq \Delta N_{p}^{\text{P50}} $$
    """
    severity = Severity.STRICT

    mid_rec_oil = project_resources.loc[
        project_resources["uncert_lvl"] == "2. Middle Value",
        ["project_id", "report_year", "rec_oil"],
    ].rename(columns={"rec_oil": "mid_rec_oil"})

    project_resources = project_resources.merge(
        mid_rec_oil, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "1. Low Value"]["rec_oil"]
        <= project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
            "mid_rec_oil"
        ]
    )
    logging.debug("%s: %s %s", re0015.__name__, invalid.sum(), severity.value)
    return "rec_oil", re0015.__name__, severity, invalid


def re0016(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0016 - Oil GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{\text{P50}} \leq \Delta N_{pn}^{\text{P10}} $$
    """
    severity = Severity.STRICT

    hgh_rec_oil = project_resources.loc[
        project_resources["uncert_lvl"] == "3. High Value",
        ["project_id", "report_year", "rec_oil"],
    ].rename(columns={"rec_oil": "hgh_rec_oil"})

    project_resources = project_resources.merge(
        hgh_rec_oil, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "rec_oil"
        ]
        <= project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "hgh_rec_oil"
        ]
    )
    logging.debug("%s: %s %s", re0016.__name__, invalid.sum(), severity.value)
    return "rec_oil", re0016.__name__, severity, invalid


def re0017(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0017 - Condensate GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{c \text{ P90}} \leq \Delta N_{pn}^{c \text{ P50}} $$
    """
    severity = Severity.STRICT

    mid_rec_con = project_resources.loc[
        project_resources["uncert_lvl"] == "2. Middle Value",
        ["project_id", "report_year", "rec_con"],
    ].rename(columns={"rec_con": "mid_rec_con"})

    project_resources = project_resources.merge(
        mid_rec_con, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "1. Low Value"]["rec_con"]
        <= project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
            "mid_rec_con"
        ]
    )
    logging.debug("%s: %s %s", re0017.__name__, invalid.sum(), severity.value)
    return "rec_con", re0017.__name__, severity, invalid


def re0018(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0018 - Condensate GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta N_{pn}^{c \text{ P50}} \leq \Delta N_{pn}^{c \text{ P10}} $$
    """
    severity = Severity.STRICT

    hgh_rec_con = project_resources.loc[
        project_resources["uncert_lvl"] == "3. High Value",
        ["project_id", "report_year", "rec_con"],
    ].rename(columns={"rec_con": "hgh_rec_con"})

    project_resources = project_resources.merge(
        hgh_rec_con, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "rec_con"
        ]
        <= project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "hgh_rec_con"
        ]
    )
    logging.debug("%s: %s %s", re0018.__name__, invalid.sum(), severity.value)
    return "rec_con", re0018.__name__, severity, invalid


def re0019(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0019 - Associated Gas GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{a \text{ P90}} \leq \Delta G_{pn}^{a \text{ P50}} $$
    """
    severity = Severity.STRICT

    mid_rec_ga = project_resources.loc[
        project_resources["uncert_lvl"] == "2. Middle Value",
        ["project_id", "report_year", "rec_ga"],
    ].rename(columns={"rec_ga": "mid_rec_ga"})

    project_resources = project_resources.merge(
        mid_rec_ga, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "1. Low Value"]["rec_ga"]
        <= project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
            "mid_rec_ga"
        ]
    )
    logging.debug("%s: %s %s", re0019.__name__, invalid.sum(), severity.value)
    return "rec_ga", re0019.__name__, severity, invalid


def re0020(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0020 - Associated Gas GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{a \text{ P50}} \leq \Delta G_{pn}^{a \text{ P10}} $$
    """
    severity = Severity.STRICT

    hgh_rec_ga = project_resources.loc[
        project_resources["uncert_lvl"] == "3. High Value",
        ["project_id", "report_year", "rec_ga"],
    ].rename(columns={"rec_ga": "hgh_rec_ga"})

    project_resources = project_resources.merge(
        hgh_rec_ga, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "rec_ga"
        ]
        <= project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "hgh_rec_ga"
        ]
    )
    logging.debug("%s: %s %s", re0020.__name__, invalid.sum(), severity.value)
    return "rec_ga", re0020.__name__, severity, invalid


def re0021(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0021 - Non Associated Gas GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{a \text{ P90}} \leq \Delta G_{pn}^{a \text{ P50}} $$
    """
    severity = Severity.STRICT

    mid_rec_gn = project_resources.loc[
        project_resources["uncert_lvl"] == "2. Middle Value",
        ["project_id", "report_year", "rec_gn"],
    ].rename(columns={"rec_gn": "mid_rec_gn"})

    project_resources = project_resources.merge(
        mid_rec_gn, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "1. Low Value"]["rec_gn"]
        <= project_resources[project_resources["uncert_lvl"] == "1. Low Value"][
            "mid_rec_gn"
        ]
    )
    logging.debug("%s: %s %s", re0021.__name__, invalid.sum(), severity.value)
    return "rec_gn", re0021.__name__, severity, invalid


def re0022(project_resources: pd.DataFrame) -> Tuple[str, str, Severity, pd.Series]:
    r"""
    ### RE0020 - Associated Gas GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U

    Severity:  `strict` :no_entry:

    The following equation must be true:

    $$\Delta G_{pn}^{a \text{ P50}} \leq \Delta G_{pn}^{a \text{ P10}} $$
    """
    severity = Severity.STRICT

    hgh_rec_gn = project_resources.loc[
        project_resources["uncert_lvl"] == "3. High Value",
        ["project_id", "report_year", "rec_gn"],
    ].rename(columns={"rec_gn": "hgh_rec_gn"})

    project_resources = project_resources.merge(
        hgh_rec_gn, on=["project_id", "report_year"], how="left"
    )

    invalid = ~(
        project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "rec_gn"
        ]
        <= project_resources[project_resources["uncert_lvl"] == "2. Middle Value"][
            "hgh_rec_gn"
        ]
    )
    logging.debug("%s: %s %s", re0022.__name__, invalid.sum(), severity.value)
    return "rec_gn", re0022.__name__, severity, invalid
