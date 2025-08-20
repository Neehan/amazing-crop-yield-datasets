"""Base processor class for loading administrative boundaries"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
from src.utils.geography import Geography


logger = logging.getLogger(__name__)


class BaseProcessor:
    """Base class for loading administrative boundaries using GADM data"""

    def __init__(
        self, country: str, admin_level: int = 2, data_dir: Optional[Path] = None
    ):
        """Initialize processor

        Args:
            country: Country ISO code (e.g., 'USA', 'ARG', 'BRA') or name
            admin_level: GADM administrative level (0=country, 1=state/province, 2=county/department/municipality)
            data_dir: Base data directory (defaults to ./data)
        """
        self.geography = Geography()
        self.country_iso = self.geography.get_country_iso_code(country)
        self.country_full_name = self.geography.get_country_full_name(country)
        self.admin_level = admin_level
        self.data_dir = Path(data_dir or "data")

        # Admin boundaries cache
        self._boundaries: Optional[gpd.GeoDataFrame] = None

        logger.info(
            f"Initialized {self.__class__.__name__} for {country} at admin level {admin_level}"
        )

    @property
    def boundaries(self) -> gpd.GeoDataFrame:
        """Get administrative boundaries, loading if needed"""
        if self._boundaries is None:
            self._boundaries = self._load_boundaries()
        return self._boundaries

    def _load_boundaries(self) -> gpd.GeoDataFrame:
        """Load administrative boundaries from GADM"""
        logger.info(
            f"Loading admin level {self.admin_level} boundaries for {self.country_iso}"
        )

        country_code = self.country_iso

        # GADM 4.1 direct download URL
        gadm_url = (
            f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{country_code}.gpkg"
        )

        logger.info(f"Loading GADM data from {gadm_url}")

        # Load the data
        boundaries = gpd.read_file(gadm_url, layer=f"ADM_ADM_{self.admin_level}")

        # Ensure CRS is WGS84
        if boundaries.crs != "EPSG:4326":
            boundaries = boundaries.to_crs("EPSG:4326")

        logger.info(f"Loaded {len(boundaries)} administrative units")
        return boundaries

    def get_admin_name(self, boundary_row) -> str:
        """Extract admin name from boundary row, handling GADM naming conventions"""
        # GADM uses NAME_0, NAME_1, NAME_2, etc.
        name_col = f"NAME_{self.admin_level}"
        if name_col in boundary_row:
            return boundary_row[name_col]

        # Fallback to common naming patterns
        for col in ["NAME", "name", "ADMIN_NAME", "admin_name"]:
            if col in boundary_row:
                return boundary_row[col]

        return f"Admin_{boundary_row.name}"
