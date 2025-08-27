"""Zip extractor for extracting daily NetCDF files from zip archives into annual files"""

import logging
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional
import xarray as xr
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ZipExtractor:
    """Extracts daily NetCDF files from zip archives and organizes into annual files"""

    def __init__(self, cache_dir: Path):
        """Initialize zip extractor

        Args:
            cache_dir: Directory to cache extracted annual files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_year_from_zip(
        self, zip_path: Path, variable: str, year: int, force_refresh: bool = False
    ) -> Path:
        """Extract daily NetCDF files from zip and organize into annual file

        Args:
            zip_path: Path to zip file containing daily NetCDF files
            variable: Weather variable name (e.g., 't2m_max', 'temp_min')
            year: Year of the data
            force_refresh: If True, recreate even if cached file exists

        Returns:
            Path to annual NetCDF file containing daily data
        """
        # Generate cache filename
        cache_filename = f"{year}_{variable}_daily.nc"
        cache_path = self.cache_dir / cache_filename

        # Return cached file if it exists and we're not forcing refresh
        if cache_path.exists() and not force_refresh:
            logger.debug(f"Using cached daily file: {cache_path}")
            return cache_path

        logger.debug(f"Extracting {variable} data for {year} from {zip_path}")

        # Extract all NetCDF files from zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            nc_files = [f for f in zip_ref.namelist() if f.endswith(".nc")]

            if not nc_files:
                raise ValueError(f"No NetCDF files found in {zip_path}")

            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract all NetCDF files
                logger.debug(
                    f"Extracting {len(nc_files)} NetCDF files to temporary directory"
                )
                for nc_file in nc_files:
                    zip_ref.extract(nc_file, temp_path)

                # Load and concatenate all daily files
                daily_datasets = []
                valid_files = []

                for nc_file in sorted(nc_files):
                    nc_path = temp_path / nc_file

                    # Extract date from filename
                    date_str = self._extract_date_from_filename(nc_file)
                    if not date_str:
                        logger.warning(
                            f"Skipping file with unrecognizable date: {nc_file}"
                        )
                        continue

                    # Parse date and check year
                    file_date = pd.to_datetime(date_str)
                    if file_date.year != year:
                        logger.debug(
                            f"Skipping file from different year: {nc_file} (year {file_date.year})"
                        )
                        continue

                    # Load dataset
                    ds = xr.open_dataset(nc_path)

                    # Add time coordinate if missing
                    if "time" not in ds.coords:
                        ds = ds.assign_coords(time=pd.to_datetime(date_str))
                        ds = ds.expand_dims("time")

                    daily_datasets.append(ds)
                    valid_files.append(nc_file)

                if not daily_datasets:
                    raise ValueError(
                        f"No valid daily files found for year {year} in {zip_path}"
                    )

                logger.debug(
                    f"Concatenating {len(daily_datasets)} daily files for {year}"
                )

                # Concatenate all datasets along time dimension
                annual_ds = xr.concat(daily_datasets, dim="time")

                # Sort by time to ensure proper order
                annual_ds = annual_ds.sortby("time")

            # Add metadata
            annual_ds.attrs.update(
                {
                    "title": f"Annual daily {variable} data for {year}",
                    "source": f"Extracted from {zip_path.name}",
                    "variable": variable,
                    "year": year,
                    "processing_date": pd.Timestamp.now().isoformat(),
                }
            )

            # Save annual dataset
            logger.debug(f"Saving annual dataset to {cache_path}")
            annual_ds.to_netcdf(cache_path)

            # Close datasets to free memory
            for ds in daily_datasets:
                ds.close()
            annual_ds.close()

        logger.debug(
            f"Successfully created annual file with {len(daily_datasets)} daily records: {cache_path}"
        )
        return cache_path

    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract date string from NetCDF filename

        Common patterns:
        - agera5_2020010100.nc -> 2020-01-01
        - 20200101_temperature.nc -> 2020-01-01
        - temp_20200101.nc -> 2020-01-01
        """
        import re

        # Pattern for YYYYMMDD format
        date_patterns = [
            r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
            r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
            r"(\d{4})\.(\d{2})\.(\d{2})",  # YYYY.MM.DD
        ]

        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                year, month, day = match.groups()
                date_str = f"{year}-{month}-{day}"
                pd.to_datetime(date_str)  # Will raise if invalid
                return date_str

        # If no date found, try to extract from position
        # AgERA5 often uses format like agera5_YYYYMMDDHH.nc
        match = re.search(r"agera5_(\d{8})", filename)
        if match:
            date_part = match.group(1)
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            date_str = f"{year}-{month}-{day}"
            pd.to_datetime(date_str)  # Validate
            return date_str

        logger.warning(f"Could not extract date from filename: {filename}")
        return None

    def process_all_zips(
        self,
        weather_dir: Path,
        year_range: tuple,
        variables: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[Path]:
        """Process all zip files in weather directory

        Args:
            weather_dir: Directory containing zip files
            year_range: Tuple of (start_year, end_year)
            variables: List of variables to process (None for all)
            force_refresh: If True, recreate all cached files

        Returns:
            List of paths to annual NetCDF files containing daily data
        """
        start_year, end_year = year_range
        zip_files = list(weather_dir.glob("*.zip"))

        if not zip_files:
            raise ValueError(f"No zip files found in {weather_dir}")

        annual_files = []

        # Filter zip files for the requested variables first to get accurate count
        relevant_zip_files = []
        for zip_path in zip_files:
            stem = zip_path.stem
            parts = stem.split("_")

            # Try to find year in filename
            year = None
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    candidate_year = int(part)
                    if start_year <= candidate_year < end_year:
                        year = candidate_year
                        break

            if year is None:
                continue

            # Extract variable name
            variable_parts = [p for p in parts if p != str(year)]
            variable = "_".join(variable_parts) if variable_parts else "unknown"

            # Skip if variables filter is specified and this variable is not in it
            if variables and variable not in variables:
                continue

            relevant_zip_files.append((zip_path, year, variable))

        # Debug: log what we found
        logger.info(
            f"Found {len(relevant_zip_files)} files for variables {variables} in range {start_year}-{end_year-1}"
        )

        # Create progress description based on variables
        if variables and len(variables) == 1:
            desc = f"  Extracting {variables[0]} years"
        else:
            desc = f"  Extracting years"

        # Add progress bar for processing zip files (years)
        for zip_path, year, variable in tqdm(
            relevant_zip_files, desc=desc, unit="file"
        ):
            logger.debug(
                f"Processing {zip_path.name} -> year={year}, variable={variable}"
            )

            annual_file = self.extract_year_from_zip(
                zip_path, variable, year, force_refresh
            )
            annual_files.append(annual_file)

        if not annual_files:
            raise ValueError(
                f"No files processed from {weather_dir} for years {start_year}-{end_year-1}"
            )

        logger.debug(f"Successfully processed {len(annual_files)} zip files")
        return annual_files

    def get_available_data(self, weather_dir: Path) -> List[tuple]:
        """Get list of available (year, variable) combinations

        Args:
            weather_dir: Directory containing zip files

        Returns:
            List of (year, variable) tuples
        """
        zip_files = list(weather_dir.glob("*.zip"))
        available_data = []

        for zip_path in zip_files:
            stem = zip_path.stem
            parts = stem.split("_")

            # Try to find year in filename
            year = None
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    break

            if year is None:
                continue

            # Extract variable name
            variable_parts = [p for p in parts if p != str(year)]
            variable = "_".join(variable_parts) if variable_parts else "unknown"

            available_data.append((year, variable))

        return sorted(available_data)
