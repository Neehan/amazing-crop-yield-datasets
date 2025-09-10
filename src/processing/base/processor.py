"""Base processor class for loading administrative boundaries"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import urllib.request

import geopandas as gpd
import pandas as pd
import xarray as xr
import requests
from tqdm import tqdm
from src.utils.geography import Geography
from src.constants import DOWNLOAD_CHUNK_SIZE


logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Base class for loading administrative boundaries using GADM data"""

    def __init__(
        self,
        country: str,
        admin_level: int,
        data_dir: Optional[Path],
        debug: bool = False,
    ):
        """Initialize processor

        Args:
            country: Country ISO code (e.g., 'USA', 'ARG', 'BRA') or name
            admin_level: GADM administrative level (0=country, 1=state/province, 2=county/department/municipality)
            data_dir: Base data directory (defaults to ./data)
            debug: Enable debug logging
        """
        self.geography = Geography()
        self.country_iso = self.geography.get_country_iso_code(country)
        self.country_full_name = self.geography.get_country_full_name(country)
        self.admin_level = admin_level
        self.data_dir = Path(data_dir or "data")
        self.debug = debug

        # Admin boundaries cache
        self._boundaries: Optional[gpd.GeoDataFrame] = None

        # Setup logging
        self._setup_logging()

        logger.info(
            f"Initialized {self.__class__.__name__} for {country} at admin level {admin_level}"
        )

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def get_intermediate_directory(self) -> Path:
        """Get intermediate directory for this country"""
        return self.data_dir / self.country_full_name.lower() / "intermediate"

    def get_processed_subdirectory(self, subdir: str) -> Path:
        """Get a subdirectory within intermediate"""
        processed_dir = self.get_intermediate_directory() / subdir
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir

    def process_with_validation(self) -> List[Path]:
        """Template method that validates config before processing"""
        config = getattr(self, "config", None)
        if config is not None:
            config.validate()
        return self.process()

    @abstractmethod
    def process(self) -> List[Path]:
        """Process data - to be implemented by subclasses"""
        pass

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

    def save_output(
        self,
        df: pd.DataFrame,
        filename: str,
        output_format: str,
        intermediate_dir: Path,
    ) -> Path:
        """Save dataframe to appropriate output format and directory

        Args:
            df: DataFrame to save
            filename: Name of the output file (including extension)
            output_format: Output format ('csv' or 'parquet')
            intermediate_dir: Base intermediate directory

        Returns:
            Path to saved file
        """
        # Create output directory based on format
        if output_format == "csv":
            output_dir = intermediate_dir / "aggregated"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / filename
            df.to_csv(output_file, index=False)

            # Check for NaN values after saving CSV
            self._check_nan_values(df, filename)
        elif output_format == "parquet":
            output_file = intermediate_dir / filename
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return output_file

    def _check_nan_values(self, df: pd.DataFrame, filename: str):
        """Check for NaN values in the dataset and log warnings"""
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            logger.warning(
                f"NaN values detected in {filename}: {total_nans} total NaN values"
            )

            # Log columns with NaN values
            cols_with_nans = nan_counts[nan_counts > 0]
            for col, count in cols_with_nans.items():
                pct_nan = (count / len(df)) * 100
                logger.warning(f"  - {col}: {count} NaN values ({pct_nan:.1f}%)")
        else:
            logger.info(
                f"Data quality check passed for {filename}: No NaN values found"
            )

    def combine_annual_files_in_memory(self, annual_files: List[Path]) -> xr.Dataset:
        """Combine annual NetCDF files in memory without saving

        Args:
            annual_files: List of annual NetCDF file paths

        Returns:
            Combined xarray Dataset
        """
        datasets = []
        for file_path in tqdm(sorted(annual_files), desc="Processing years"):
            ds = xr.open_dataset(file_path)
            datasets.append(ds)

        # Concatenate along time dimension
        combined_ds = xr.concat(datasets, dim="time")
        combined_ds = combined_ds.sortby("time")
        return combined_ds

    def check_file_exists(self, file_path: Path) -> bool:
        """Check if a file exists and log the result"""
        exists = file_path.exists()
        if not exists:
            logger.debug(f"File not found: {file_path}")
        return exists

    def get_csv_files_by_pattern(self, csv_dir: Path, pattern: str) -> List[Path]:
        """Get CSV files matching a pattern, with existence check"""
        csv_dir.mkdir(parents=True, exist_ok=True)

        files = list(csv_dir.glob(pattern))
        existing_files = [f for f in files if self.check_file_exists(f)]

        if not existing_files:
            logger.warning(f"No files found matching pattern: {pattern} in {csv_dir}")

        return existing_files
