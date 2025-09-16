"""
Models and configuration for Argentina crop yield data processing.

This module contains mappings between Spanish crop names and standardized English names,
as well as data quality thresholds for filtering departments with insufficient data.
"""

from typing import Dict, Set
from src.crop_yield.base.constants import (
    AREA_PLANTED_COLUMN,
    AREA_HARVESTED_COLUMN,
    PRODUCTION_COLUMN,
    YIELD_COLUMN,
)

# Mapping from Spanish crop names (as they appear in the data) to standardized English names
CROP_NAME_MAPPING: Dict[str, str] = {
    "Trigo total": "wheat",
    "Soja total": "soybean",
    "Maíz": "corn",
    "Girasol": "sunflower",
    "Soja 1ra": "soybean1",
    "Soja 2da": "soybean2",
}

# Set of all supported crops for validation
SUPPORTED_CROPS: Set[str] = set(CROP_NAME_MAPPING.keys())


# Column names in the raw data
RAW_DATA_COLUMNS = {
    "crop": "Cultivo",
    "season": "Campaña",
    "province": "Provincia",
    "department": "Departamento",
    "province_id": "idProvincia",
    "department_id": "idDepartamento",
    YIELD_COLUMN: "Rendimiento",
    AREA_PLANTED_COLUMN: "Sup. Sembrada",
    AREA_HARVESTED_COLUMN: "Sup. Cosechada",
    PRODUCTION_COLUMN: "Producción",
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
