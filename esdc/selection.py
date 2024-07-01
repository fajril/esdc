"""
Enum classes for API versions, table names, and file types.
"""
from enum import Enum

class ApiVer(Enum):
    """
    Enum for API versions.
    
    Attributes:
        V1 (int): API version 1.
        V2 (str): API version 2.
    """
    V1 = 1
    V2 = "api/v2"


class TableName(Enum):
    """
    Enum for table names.
    
    Attributes:
        PROJECT_RESOURCES (str): Table name for project resources.
        PROJECt_TIMESERIES (str): Table name for project timeseries.
    """
    PROJECT_RESOURCES = "project_resources"
    FIELD_RESOURCES = "field_resources"
    WA_RESOURCES = "wa_resources"
    NKRI_RESOURCES = "nkri_resources"
    PROJECT_TIMESERIES = "project_timeseries"


class FileType(Enum):
    """
    Enum for file types.
    
    Attributes:
        CSV (str): CSV file type.
        JSON (str): JSON file type.
        ZIP (str): ZIP file type.
    """
    CSV = "csv"
    JSON = "json"
    ZIP = "zip"


