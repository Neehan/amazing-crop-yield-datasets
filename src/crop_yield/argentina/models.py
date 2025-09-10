"""
Models and configuration for Argentina crop yield data processing.

This module contains mappings between Spanish crop names and standardized English names,
as well as data quality thresholds for filtering departments with insufficient data.
"""

from typing import Dict, Set

# Mapping from Spanish crop names (as they appear in the data) to standardized English names
CROP_NAME_MAPPING: Dict[str, str] = {
    "Trigo total": "wheat",
    "Soja total": "soybean",
    "Maíz": "corn",
    "Girasol": "sunflower",
}

# Set of all supported crops for validation
SUPPORTED_CROPS: Set[str] = set(CROP_NAME_MAPPING.keys())

# Data quality thresholds
DATA_QUALITY_THRESHOLD = 0.25  # 25% of data must be present (75% missing allowed)
EVALUATION_YEARS = 20  # Evaluate data quality over the last 20 years

# Column names in the raw data
RAW_DATA_COLUMNS = {
    "crop": "Cultivo",
    "season": "Campaña",
    "province": "Provincia",
    "department": "Departamento",
    "province_id": "idProvincia",
    "department_id": "idDepartamento",
    "yield": "Rendimiento",
}

# Output CSV column names
OUTPUT_COLUMNS = [
    "country",
    "admin_level_1",
    "admin_level_2",
    "year",
    # crop yield columns will be added dynamically: {crop_name}_yield
]

# Constants
COUNTRY_NAME = "Argentina"
