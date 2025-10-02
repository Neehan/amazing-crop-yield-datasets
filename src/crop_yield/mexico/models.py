"""
Models and configuration for Mexico crop yield data processing.

This module contains mappings between Spanish crop names and standardized English names,
as well as SIAP crop codes and data quality thresholds.
"""

from typing import Dict, Set

# SIAP crop codes mapping (actual crop codes)
CROP_CODES: Dict[str, str] = {
    "sugarcane": "70",  # Caña de azúcar
    "barley": "77",  # Cebada grano
    "beans": "152",  # Frijol
    "corn": "225",  # Maíz grano
    "sorghum": "374",  # Sorghum
    "soybean": "375",  # Soya
    "tomato": "389",  # Tomate rojo (jitomate)
    "wheat": "395",  # Trigo grano
    # Additional crops will be added as we discover their codes
}


# Function to detect irrigated crop variants
def is_irrigated_crop(crop_name: str) -> bool:
    """Check if a crop name ends with '_irrigated'."""
    return crop_name.endswith("_irrigated")


def is_rainfed_crop(crop_name: str) -> bool:
    """Check if a crop name ends with '_rainfed'."""
    return crop_name.endswith("_rainfed")


def get_base_crop_name(crop_name: str) -> str:
    """Extract base crop name from irrigated or rainfed variant (e.g., 'corn_irrigated' -> 'corn', 'beans_rainfed' -> 'beans')."""
    if is_irrigated_crop(crop_name):
        return crop_name[:-10]  # Remove '_irrigated' suffix
    if is_rainfed_crop(crop_name):
        return crop_name[:-8]  # Remove '_rainfed' suffix
    return crop_name


def get_irrigated_crop_name(base_crop: str) -> str:
    """Get irrigated variant name from base crop (e.g., 'corn' -> 'corn_irrigated')."""
    return f"{base_crop}_irrigated"


def get_rainfed_crop_name(base_crop: str) -> str:
    """Get rainfed variant name from base crop (e.g., 'corn' -> 'corn_rainfed')."""
    return f"{base_crop}_rainfed"


# SIAP crop type code (same for all field crops)
CROP_TYPE_CODE = "5"  # Field crop type

# Mapping from Spanish crop names (as they appear in the data) to standardized English names
CROP_NAME_MAPPING: Dict[str, str] = {
    "Maíz grano": "corn",
    "Soya": "soybean",
    "Trigo grano": "wheat",
    "Sorgo grano": "sorghum",
    "Caña de azúcar": "sugarcane",
    "Tomate rojo": "tomato",
    "Frijol": "beans",
    "Cebada grano": "barley",
}

# Set of all supported crops for validation
SUPPORTED_CROPS: Set[str] = set(CROP_NAME_MAPPING.keys())

# Column names in the raw HTML data (based on table structure)
RAW_DATA_COLUMNS = {
    "state": "Entidad",
    "municipality": "Municipio",
    "area_planted": "Sembrada",  # Superficie Sembrada (ha)
    "area_harvested": "Cosechada",  # Superficie Cosechada (ha)
    "area_damaged": "Siniestrada",  # Superficie Siniestrada (ha)
    "production": "Producción",  # Producción (ton)
    "yield": "Rendimiento",  # Rendimiento (udm/ha)
    "price": "PMR",  # Precio Medio Rural ($/udm)
    "value": "Valor Producción",  # Valor Producción (miles de Pesos)
}

# Output CSV column names (matching Brazil format exactly)
OUTPUT_COLUMNS = [
    "country",
    "admin_level_1",
    "admin_level_2",
    "year",
    # crop yield columns will be added dynamically: {crop_name}_yield
    "area_planted",
    "area_harvested",
    "production",
]

# Constants
COUNTRY_NAME = "Mexico"

# API configuration
API_BASE_URL = "https://nube.agricultura.gob.mx/cierre_agricola/"

# Request parameters structure
API_PARAMS = {
    "nivel_municipio": "1",  # Position 0: Nivel Municipio
    "modalidad_riego_temporal": "3",  # Position 3: Modalidad (3 = Riego + Temporal)
    "modalidad_riego_only": "1",  # Position 3: Modalidad (1 = Riego only)
    "modalidad_temporal_only": "2",  # Position 3: Modalidad (2 = Temporal only / rainfed)
    "ciclo_todos": "0",  # Position 4: All cycles
    "estado_placeholder": "--",  # Position 5: Estado placeholder
    "municipio_placeholder": "--",  # Position 6: Municipio placeholder
    "municipio_code": "200201",  # Position 8: Municipality code
    "additional_params": [
        "0",
        "1",
        "0",
        "0",
        "0",
    ],  # Positions 9-13: Additional parameters
}
