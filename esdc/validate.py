from typing import Tuple, List
import logging
import numpy as np
import pandas as pd

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
            "re0007": re0.re0007,
            "re0008": re0.re0008,
            "re0009": re0.re0009,
            "re0010": re0.re0010,
            "re0011": re0.re0011,
            "re0012": re0.re0012,
        }

    def run(self) -> List[Tuple[str, str, Severity, pd.Series]]:
        results = [
            func(self.project_resources) for _, func in self.rules0.items()
        ]
        return self._val_results_transform(results)

    def _val_results_transform(self, results):
        validation_volumetric = self.project_resources[self.volumetric_columns].copy()
        validation_volumetric = validation_volumetric.astype(str)
        validation_volumetric.loc[:, self.volumetric_columns[7:]] = ""

        for result in results:
            if result[3].any():
                validation_volumetric.loc[result[3], result[0]] += f"{result[1]};"
            validation_volumetric = validation_volumetric.mask(
                validation_volumetric == "", np.nan
            )

        validation_volumetric.dropna(
            subset=self.volumetric_columns[7:], how="all", inplace=True
        )
        validation_volumetric.dropna(axis=1, how="all", inplace=True)
        return validation_volumetric
