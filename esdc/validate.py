from typing import Tuple, List
from enum import Enum
import pandas as pd

from esdc import re0

class Severity(Enum):
    STRICT = 1
    WARNING = 2
    INFO = 3

class RuleEngine:
    def __init__(self, project_resources: pd.DataFrame):
        self.project_resources = project_resources

    def run(self) -> List[Tuple[str, str, Severity, pd.Series]]:
        result = [
            re0.re0007(self.project_resources),
            re0.re0008(self.project_resources),
            re0.re0009(self.project_resources),
            re0.re0010(self.project_resources),
        ]
        return result