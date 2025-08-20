"""Zip extractor for combining daily NetCDF files from zip archives"""

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
    """Extracts and combines daily NetCDF files from zip archives"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize zip extractor

        Args:
            cache_dir: Directory to cache extracted and combined files
        """
        self.cache_dir = Path(cache_dir or "data/cache/extracted")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_combine_year(
        self, zip_path: Path, variable: str, year: int, force_refresh: bool = False
    ) -> Path:
        """Extract daily NetCDF files from zip and combine into annual file

        Args:
            zip_path: Path to zip file containing daily NetCDF files
            variable: Weather variable name (e.g., 't2m_max', 'temp_min')
            year: Year of the data
            force_refresh: If True, recreate even if cached file exists

        Returns:
            Path to combined annual NetCDF file
        """
        # Generate cache filename
        cache_filename = f"{year}_{variable}_combined.nc"
        cache_path = self.cache_dir / cache_filename

        # Return cached file if it exists and we're not forcing refresh
        if cache_path.exists() and not force_refresh:
            logger.debug(f"Using cached combined file: {cache_path}")
            return cache_path

        logger.info(f"Extracting and combining {zip_path.name} -> {cache_filename}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract zip file
            logger.debug(f"Extracting zip to temporary directory: {temp_path}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Find all NetCDF files in extracted directory
            nc_files = list(temp_path.rglob("*.nc"))
            if not nc_files:
                raise ValueError(f"No NetCDF files found in {zip_path}")

            logger.debug(f"Found {len(nc_files)} NetCDF files to combine")

            # Sort files by date (assuming filename contains date info)
            nc_files.sort()

            # Load and combine all daily files
            daily_datasets = []
            for nc_file in nc_files:
                try:
                    ds = xr.open_dataset(nc_file)
                    # Add time dimension if missing (extract from filename)
                    if "time" not in ds.dims:
                        date_str = self._extract_date_from_filename(nc_file.name)
                        if date_str:
                            ds = ds.expand_dims("time")
                            ds = ds.assign_coords(time=[pd.to_datetime(date_str)])
                    daily_datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load {nc_file}: {e}")
                    continue

            if not daily_datasets:
                raise ValueError(
                    f"No valid NetCDF files could be loaded from {zip_path}"
                )

            # Combine along time dimension
            logger.debug(f"Combining {len(daily_datasets)} daily datasets")
            combined_ds = xr.concat(daily_datasets, dim="time")

            # Sort by time to ensure chronological order
            combined_ds = combined_ds.sortby("time")

            # Add metadata
            combined_ds.attrs.update(
                {
                    "title": f"Combined daily {variable} data for {year}",
                    "source": f"Extracted from {zip_path.name}",
                    "variable": variable,
                    "year": year,
                    "processing_date": pd.Timestamp.now().isoformat(),
                }
            )

            # Save combined dataset
            logger.debug(f"Saving combined dataset to {cache_path}")
            combined_ds.to_netcdf(cache_path)

            # Close datasets to free memory
            for ds in daily_datasets:
                ds.close()
            combined_ds.close()

        logger.info(
            f"Successfully combined {len(daily_datasets)} daily files into {cache_path}"
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
                try:
                    # Validate date
                    date_str = f"{year}-{month}-{day}"
                    pd.to_datetime(date_str)  # Will raise if invalid
                    return date_str
                except:
                    continue

        # If no date found, try to extract from position
        # AgERA5 often uses format like agera5_YYYYMMDDHH.nc
        match = re.search(r"agera5_(\d{8})", filename)
        if match:
            date_part = match.group(1)
            try:
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                date_str = f"{year}-{month}-{day}"
                pd.to_datetime(date_str)  # Validate
                return date_str
            except:
                pass

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
            List of paths to combined annual NetCDF files
        """
        start_year, end_year = year_range
        zip_files = list(weather_dir.glob("*.zip"))

        if not zip_files:
            raise ValueError(f"No zip files found in {weather_dir}")

        combined_files = []

        for zip_path in zip_files:
            # Extract year and variable from filename (e.g., "2020_t2m_max.zip")
            try:
                filename_parts = zip_path.stem.split("_")
                file_year = int(filename_parts[0])
                file_var = "_".join(filename_parts[1:])

                # Filter by year range
                if file_year < start_year or file_year >= end_year:
                    continue

                # Filter by variables
                if variables and file_var not in variables:
                    continue

                # Extract and combine
                combined_path = self.extract_and_combine_year(
                    zip_path, file_var, file_year, force_refresh
                )
                combined_files.append(combined_path)

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse filename {zip_path.name}: {e}")
                continue

        logger.info(
            f"Processed {len(combined_files)} zip files into combined annual files"
        )
        return combined_files

    def process_all_zips_to_single_file(
        self,
        weather_dir: Path,
        output_dir: Path,
        year_range: tuple,
        variables: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Path:
        """Process all zip files directly into single multi-year NetCDF file

        Args:
            weather_dir: Directory containing zip files
            output_dir: Directory to save the combined file
            year_range: Tuple of (start_year, end_year)
            variables: List of variables to process (None for all)
            force_refresh: If True, recreate even if combined file exists

        Returns:
            Path to combined multi-year NetCDF file
        """
        start_year, end_year = year_range

        # Generate output filename
        var_suffix = "_".join(variables) if variables else "all_vars"
        output_filename = f"weather_{start_year}_{end_year-1}_{var_suffix}_combined.nc"
        output_path = output_dir / output_filename

        # Return existing file if not forcing refresh
        if output_path.exists() and not force_refresh:
            logger.info(f"Using existing combined file: {output_path}")
            return output_path

        # Find zip files to process
        zip_files = list(weather_dir.glob("*.zip"))
        if not zip_files:
            raise ValueError(f"No zip files found in {weather_dir}")

        # Filter zip files by year range and variables
        valid_zips = []
        for zip_path in zip_files:
            try:
                filename_parts = zip_path.stem.split("_")
                file_year = int(filename_parts[0])
                file_var = "_".join(filename_parts[1:])

                # Filter by year range and variables
                if file_year < start_year or file_year >= end_year:
                    continue
                if variables and file_var not in variables:
                    continue

                valid_zips.append((zip_path, file_year, file_var))
            except (ValueError, IndexError):
                continue

        if not valid_zips:
            raise ValueError("No valid zip files found for specified criteria")

        logger.info(
            f"Processing {len(valid_zips)} zip files directly into combined file..."
        )

        # Process all zips and combine datasets
        all_datasets = []

        for zip_path, year, variable in tqdm(
            valid_zips, desc="Extracting and combining zips"
        ):
            logger.debug(f"Processing {zip_path.name}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract zip file
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)

                # Find NetCDF files
                nc_files = list(temp_path.rglob("*.nc"))
                if not nc_files:
                    logger.warning(f"No NetCDF files found in {zip_path}")
                    continue

                # Load and combine daily files for this year
                daily_datasets = []
                for nc_file in nc_files:
                    try:
                        ds = xr.open_dataset(nc_file)
                        if "time" not in ds.dims:
                            date_str = self._extract_date_from_filename(nc_file.name)
                            if date_str:
                                ds = ds.expand_dims("time")
                                ds = ds.assign_coords(time=[pd.to_datetime(date_str)])
                        daily_datasets.append(ds)
                    except Exception as e:
                        logger.warning(f"Could not load {nc_file}: {e}")

                if daily_datasets:
                    # Combine daily files for this year
                    year_ds = xr.concat(daily_datasets, dim="time").sortby("time")

                    # Add year coordinate
                    year_ds = year_ds.assign_coords(
                        year=("time", [year] * len(year_ds.time))
                    )
                    all_datasets.append(year_ds)

        if not all_datasets:
            raise ValueError("No datasets could be created from zip files")

        # Combine all years
        logger.info("Combining all years into single dataset...")
        combined_dataset = xr.concat(all_datasets, dim="time")

        # Save to processed directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving combined dataset to {output_path}")
        combined_dataset.to_netcdf(output_path)

        # Clean up
        for ds in all_datasets:
            ds.close()
        combined_dataset.close()

        logger.info(f"Successfully created combined multi-year file: {output_path}")
        return output_path
