"""Constants for integration tests."""

import re

# Matches project-resources endpoint WITHOUT report-year parameter
PROJECT_RESOURCES_URL = re.compile(
    r"https://esdc\.skkmigas\.go\.id/api/v2/project-resources\?verbose=3&output=.*"
)

# Matches any project-resources endpoint (with or without report-year)
PROJECT_RESOURCES_ANY_URL = re.compile(
    r"https://esdc\.skkmigas\.go\.id/api/v2/project-resources\?.*"
)

# Matches any project-timeseries endpoint (with or without report-year)
# (Legacy wide pattern kept for compatibility)
PROJECT_TIMESERIES_URL = re.compile(
    r"https://esdc\.skkmigas\.go\.id/api/v2/project-timeseries.*"
)
