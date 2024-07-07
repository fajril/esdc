from typing import Tuple, List
import logging
import numpy as np
import pandas as pd

from rich.progress import track
from .selection import Severity
from . import re0


class RuleEngine:
    def __init__(self, project_resources: pd.DataFrame):
        self.project_resources = project_resources
        self.volumetric_columns = [
            "report_year",
            "wk_name",
            "field_id",
            "field_name",
            "project_id",
            "project_name",
            "uncert_lvl",
            "rec_oil",
            "rec_con",
            "rec_ga",
            "rec_gn",
            "res_oil",
            "res_con",
            "res_ga",
            "res_gn",
            "prj_ioip",
            "prj_igip",
        ]
        self.rules0 = {
            re0.re0007.__name__: re0.re0007,
            re0.re0008.__name__: re0.re0008,
            re0.re0009.__name__: re0.re0009,
            re0.re0010.__name__: re0.re0010,
            re0.re0011.__name__: re0.re0011,
            re0.re0012.__name__: re0.re0012,
            re0.re0013.__name__: re0.re0013,
            re0.re0014.__name__: re0.re0014,
            re0.re0015.__name__: re0.re0015,
            re0.re0016.__name__: re0.re0016,
            re0.re0017.__name__: re0.re0017,
            re0.re0018.__name__: re0.re0018,
            re0.re0019.__name__: re0.re0019,
            re0.re0020.__name__: re0.re0020,
            re0.re0021.__name__: re0.re0021,
            re0.re0022.__name__: re0.re0022,
        }

    def run(self) -> List[Tuple[str, str, Severity, pd.Series]]:
        """
        Initiates the validation process.

        This method runs all validation rules.
        The results of each validation rule are collected and 
        transformed into a standardized format.

        Returns:
            A list of tuples, where each tuple contains the following information:
                - rule_name (str): The name of the validation rule.
                - message (str): A brief message describing the validation result.
                - severity (Severity): The severity level of the validation result 
                  (e.g., error, warning, info).
                - series (pd.Series): A pandas Series containing the validation result data.
        """
        logging.info("Initiate validation process.")
        if logging.root.getEffectiveLevel() == 10:
            iter_rules0 = self.rules0.items()
        else:
            iter_rules0 = track(self.rules0.items())
        results = [func(self.project_resources) for _, func in iter_rules0]

        return self._val_results_transform(results)

    def _val_results_transform(self, results):
        validation_volumetric = self.project_resources[self.volumetric_columns].copy()
        validation_volumetric = validation_volumetric.astype(str)
        validation_volumetric.loc[:, self.volumetric_columns[7:]] = ""

        for result in results:
            if result[3].any():
                idx = result[3].loc[result[3]].index
                validation_volumetric.loc[idx, result[0]] += f"{result[1].upper()};"
        validation_volumetric = validation_volumetric.mask(
            validation_volumetric == "", np.nan
        )
        validation_volumetric.dropna(
            subset=self.volumetric_columns[7:], how="all", inplace=True
        )
        validation_volumetric.dropna(axis=1, how="all", inplace=True)
        return validation_volumetric
