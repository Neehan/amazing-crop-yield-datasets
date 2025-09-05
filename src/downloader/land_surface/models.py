"""Land surface data models and enums for Google Earth Engine"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List


class LandSurfaceVariable(Enum):
    """Available land surface variables from ERA5-Land via Google Earth Engine"""

    LAI_HIGH = ("leaf_area_index_high_vegetation", "lai_high")
    LAI_LOW = ("leaf_area_index_low_vegetation", "lai_low")
    NDVI = ("NDVI", "ndvi")

    def __init__(self, variable: str, key_suffix: str):
        self.variable = variable  # GEE band name
        self.key_suffix = key_suffix  # File naming suffix

    @property
    def key(self):
        return self.name.lower()  # Use enum name as key


@dataclass
class GEEConfig:
    """Configuration for Google Earth Engine ERA5-Land downloads"""

    dataset: str = "ECMWF/ERA5_LAND/DAILY_AGGR"
    scale_meters: int = 11132  # ~0.1° pixels
    crs: str = "EPSG:4326"
    max_pixels: int = int(1e13)

    # Country boundaries dataset
    boundaries_dataset: str = "FAO/GAUL/2015/level0"
    country_property: str = "ADM0_NAME"

    # Export settings
    export_format: str = "GeoTIFF"
    file_per_band: bool = False  # Export all bands in one file

    # NDVI dataset configurations
    ndvi_avhrr_dataset: str = "NOAA/CDR/AVHRR/NDVI/V5"  # 1982-2013
    ndvi_viirs_dataset: str = "NOAA/CDR/VIIRS/NDVI/V1"  # 2014+
    ndvi_transition_year: int = 2014

    def get_ndvi_dataset(self, year: int) -> str:
        """Get appropriate NDVI dataset based on year"""
        if year < self.ndvi_transition_year:
            return self.ndvi_avhrr_dataset
        else:
            return self.ndvi_viirs_dataset
