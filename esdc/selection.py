"""Enum classes for API versions, table names, and file types."""

from enum import Enum


class ApiVer(Enum):
    """
    Enum for API versions.

    Attributes:
        V1 (str): API version 1 path.
        V2 (str): API version 2 path.
    """

    V1 = "/api/v1"
    V2 = "/api/v2"


class TableName(Enum):
    """
    Enum for table names.

    Attributes:
        PROJECT_RESOURCES (str): Table name for project resources.
        PROJECT_TIMESERIES (str): Table name for project timeseries.
    """

    PROJECT_RESOURCES = "project_resources"
    FIELD_RESOURCES = "field_resources"
    WA_RESOURCES = "wa_resources"
    NKRI_RESOURCES = "nkri_resources"
    PROJECT_TIMESERIES = "project_timeseries"


class PodApiTable(str, Enum):
    """
    Enum for the five canonical POD input tables served by the v1 endpoints.

    Attributes:
        MASTER_DATA_PROJECT (str): Canonical table ``m_project``.
        MASTER_DATA_POD (str): Canonical table ``m_pod``.
        POD_PROJECT (str): Canonical table ``pod_project``.
        POD_MONITORING (str): Canonical table ``pod_monitoring``.
        POD_FORECAST (str): Canonical table ``pod_forecast``.

    Notes:
    -----
    The values are the canonical input table names the POD contract and adapter
    use; the endpoint slugs are configuration, not table identity.
    """

    MASTER_DATA_PROJECT = "m_project"
    MASTER_DATA_POD = "m_pod"
    POD_PROJECT = "pod_project"
    POD_MONITORING = "pod_monitoring"
    POD_FORECAST = "pod_forecast"


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


class Severity(Enum):
    """Enum for severity levels.

    Attributes:
        STRICT (str): Strict severity level.
        WARNING (str): Warning severity level.
        INFO (str): Info severity level.
    """

    STRICT = "strict"
    WARNING = "warning"
    INFO = "info"
