"""Base processor class for loading administrative boundaries"""

import logging
from pathlib import Path
from typing import Optional
import urllib.request

import geopandas as gpd
import requests
from tqdm import tqdm
from src.utils.geography import Geography
from src.constants import DOWNLOAD_CHUNK_SIZE


logger = logging.getLogger(__name__)


class BaseProcessor:
    """Base class for loading administrative boundaries using GADM data"""

    def __init__(self, country: str, admin_level: int, data_dir: Optional[Path]):
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
        country_name = self.country_full_name.lower()

        # Check for cached GADM file in country-specific gadm subdirectory
        gadm_dir = self.data_dir / country_name / "gadm"
        gadm_dir.mkdir(parents=True, exist_ok=True)
        cached_file = gadm_dir / f"gadm41_{country_code}.gpkg"

        if cached_file.exists():
            logger.info(f"Loading cached GADM file: {cached_file}")
        else:
            # GADM 4.1 direct download URL
            gadm_url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{country_code}.gpkg"

            logger.info(f"Downloading GADM data from {gadm_url}")
            logger.info(f"Caching to: {cached_file}")

            # Download and cache the file with progress bar
            self._download_with_progress(gadm_url, cached_file, country_code)
            logger.info(f"Download complete: {cached_file}")

        # Load boundaries from cached file
        boundaries = gpd.read_file(cached_file, layer=f"ADM_ADM_{self.admin_level}")

        logger.info(
            f"Step 2: Processing {len(boundaries)} administrative boundaries..."
        )

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

    def _download_with_progress(self, url: str, output_path: Path, country_code: str):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading GADM {country_code}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                file.write(chunk)
                pbar.update(len(chunk))
