"""CCI cropland mask data processor"""

import io
import logging
import zipfile
from pathlib import Path
from typing import List

import xarray as xr
import numpy as np
from tqdm import tqdm

from src.processing.base.processor import BaseProcessor
from src.processing.management.cci_cropland_mask.config import CCICroplandMaskConfig
from src.processing.management.cci_cropland_mask.ml_imputation_processor import (
    MLImputationProcessor,
)

logger = logging.getLogger(__name__)


class CCICroplandMaskProcessor(BaseProcessor):
    """CCI cropland mask processor that creates cropland and irrigated masks"""

    def __init__(self, config: CCICroplandMaskConfig):
        """Initialize CCI cropland mask processor with configuration"""
        super().__init__(
            config.country, config.admin_level, config.data_dir, config.debug
        )
        self.config = config

        logger.info(f"CCICroplandMaskProcessor initialized for {config.country}")

    def process(self) -> List[Path]:
        """Process CCI cropland mask data according to configuration"""
        logger.info(
            f"Processing CCI cropland mask data for {self.country_full_name} ({self.config.start_year}-{self.config.end_year})"
        )

        # Get CCI cropland mask directory and intermediate output directory
        cci_cropland_mask_dir = self.config.get_cci_cropland_mask_directory()
        intermediate_dir = self.get_intermediate_directory()
        cci_cropland_mask_processed_dir = self.get_processed_subdirectory(
            "cci_cropland_mask"
        )

        output_files = []

        # Process available years (1992-2022) from zip files
        available_years = []
        missing_years = []

        for year in range(self.config.start_year, self.config.end_year):
            zip_file = cci_cropland_mask_dir / f"{year}_cropland_mask.zip"
            if zip_file.exists():
                available_years.append(year)
            else:
                missing_years.append(year)

        # Process available years
        for year in tqdm(available_years, desc="Processing CCI cropland mask"):
            logger.debug(f"Processing CCI cropland mask for year {year}")
            zip_file = cci_cropland_mask_dir / f"{year}_cropland_mask.zip"
            year_output_files = self._process_year(
                zip_file, year, cci_cropland_mask_processed_dir
            )
            output_files.extend(year_output_files)

        # Run ML imputation for missing years
        if missing_years:
            logger.info(
                f"Running ML imputation for {len(missing_years)} missing years: {missing_years}"
            )
            try:
                ml_processor = MLImputationProcessor(self.config)
                ml_output_files = ml_processor.process(missing_years)
                output_files.extend(ml_output_files)
                logger.info(f"ML imputation completed. Added {len(ml_output_files)} files.")
            except Exception as e:
                logger.error(f"ML imputation failed: {e}")
                raise

        return output_files

    def _process_year(self, zip_file: Path, year: int, output_dir: Path) -> List[Path]:
        """Process CCI cropland mask data for a single year"""
        output_files = []

        # Extract NetCDF to temp file
        temp_nc_file = output_dir / f"temp_{year}.nc"

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            # Get the NetCDF filename from the zip
            nc_filename = None
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith(".nc"):
                    nc_filename = file_info.filename
                    break

            if not nc_filename:
                raise ValueError(f"No NetCDF file found in {zip_file}")

            # Extract NetCDF file
            with zip_ref.open(nc_filename) as nc_file:
                with open(temp_nc_file, "wb") as temp_file:
                    temp_file.write(nc_file.read())

        logger.debug(f"Processing NetCDF file: {nc_filename}")

        # Read the NetCDF file
        ds = xr.open_dataset(temp_nc_file)
        lccs = ds["lccs_class"].isel(time=0)

        # Create cropland mask (union of classes 10, 11, 12, 20, 30)
        cropland_mask = self._create_cropland_mask(lccs)

        # Create irrigated mask (class 20 only)
        irrigated_mask = self._create_irrigated_mask(lccs)

        # Save masks as NetCDF files
        cropland_file = output_dir / f"cci_cropland_mask_{year}.nc"
        irrigated_file = output_dir / f"cci_cropland_mask_irrigated_{year}.nc"

        self._save_mask_as_netcdf(
            cropland_mask, cropland_file, f"CCI cropland mask for {year}"
        )
        self._save_mask_as_netcdf(
            irrigated_mask, irrigated_file, f"CCI irrigated cropland mask for {year}"
        )

        # Clean up temp file
        temp_nc_file.unlink()

        output_files.extend([cropland_file, irrigated_file])
        logger.debug(
            f"Created CCI cropland masks for year {year}: {cropland_file.name}, {irrigated_file.name}"
        )

        return output_files

    def _create_cropland_mask(self, lccs: xr.DataArray) -> xr.DataArray:
        """Create CCI cropland mask (union of all cropland classes)"""
        # Create binary mask where 1 = cropland, 0 = not cropland
        cropland_mask = xr.where(lccs.isin(self.config.CROPLAND_CLASSES), 1, 0)

        # Add attributes
        cropland_mask.attrs = {
            "long_name": "CCI cropland mask",
            "description": f"Binary mask for all cropland classes {self.config.CROPLAND_CLASSES}",
            "classes_included": "10: cropland_rainfed, 11: cropland_rainfed_herbaceous_cover, 12: cropland_rainfed_tree_or_shrub_cover, 20: cropland_irrigated, 30: mosaic_cropland",
            "values": "1 = cropland, 0 = not cropland",
        }

        return cropland_mask

    def _create_irrigated_mask(self, lccs: xr.DataArray) -> xr.DataArray:
        """Create CCI irrigated cropland mask (irrigated class only)"""
        # Create binary mask where 1 = irrigated cropland, 0 = not irrigated cropland
        irrigated_mask = xr.where(lccs == self.config.IRRIGATED_CLASS, 1, 0)

        # Add attributes
        irrigated_mask.attrs = {
            "long_name": "CCI irrigated cropland mask",
            "description": f"Binary mask for irrigated cropland (class {self.config.IRRIGATED_CLASS})",
            "class_included": f"{self.config.IRRIGATED_CLASS}: cropland_irrigated",
            "values": "1 = irrigated cropland, 0 = not irrigated cropland",
        }

        return irrigated_mask

    def _save_mask_as_netcdf(self, mask: xr.DataArray, output_file: Path, title: str):
        """Save mask as NetCDF file with proper metadata"""
        # Create dataset with the mask
        ds = xr.Dataset({"cci_cropland_mask": mask})

        # Add global attributes
        ds.attrs = {
            "title": title,
            "source": "CCI Land Cover v2.1.1",
            "institution": "Copernicus Climate Data Store",
            "history": f"Created by CCICroplandMaskProcessor on {Path(__file__).name}",
            "conventions": "CF-1.6",
        }

        # Save to NetCDF
        ds.to_netcdf(output_file)
        logger.debug(f"Saved mask to: {output_file}")
