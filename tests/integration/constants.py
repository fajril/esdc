"""Constants for integration tests."""

import re

PROJECT_RESOURCES_URL = re.compile(
    r"https://esdc\.skkmigas\.go\.id/api/v2/project-resources.*"
)
PROJECT_TIMESERIES_URL = re.compile(
    r"https://esdc\.skkmigas\.go\.id/api/v2/project-timeseries.*"
)
