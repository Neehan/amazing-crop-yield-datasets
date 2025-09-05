"""Constants used throughout the project"""

from pathlib import Path

# Data directory
DATA_DIR = "data"

# Administrative levels
ADMIN_LEVEL_COUNTRY = 0
ADMIN_LEVEL_STATE = 1
ADMIN_LEVEL_COUNTY = 2

# Weather data
WEATHER_START_YEAR_MIN = 1979  # AgERA5 data availability
WEATHER_WEEKS_PER_YEAR = 52

# Soil data
SOIL_RESOLUTION_DEGREES = 0.1  # Resolution for soil data downloads

# File processing
CHUNK_SIZE_TIME_PROCESSING = 100  # Time steps to process at once
DOWNLOAD_CHUNK_SIZE = 8192  # Bytes for file downloads

# Output formats
OUTPUT_FORMAT_CSV = "csv"
OUTPUT_FORMAT_PARQUET = "parquet"

# Default values for CLI (only place defaults are allowed)
DEFAULT_ADMIN_LEVEL = ADMIN_LEVEL_COUNTY
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_CSV
DEFAULT_START_YEAR = WEATHER_START_YEAR_MIN
