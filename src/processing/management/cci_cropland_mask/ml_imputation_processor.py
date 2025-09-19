"""Matrix factorization processor for extending CCI cropland mask using HYDE data"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import xarray as xr
from tqdm import tqdm

from src.processing.base.processor import BaseProcessor
from src.processing.management.cci_cropland_mask.config import CCICroplandMaskConfig
from src.processing.management.cci_cropland_mask.matrix_factorization_models import (
    MatrixFactorizationProcessor,
    MatrixFactorizationConstants,
)

logger = logging.getLogger(__name__)


class MLImputationProcessor(BaseProcessor):
    """Processor for matrix factorization imputation of CCI cropland mask using HYDE data"""

    def __init__(self, config: CCICroplandMaskConfig):
        """Initialize matrix factorization imputation processor"""
        super().__init__(
            config.country, config.admin_level, config.data_dir, config.debug
        )
        self.config = config

        # Data containers
        self.hyde_cropland: xr.Dataset | None = None
        self.cci_data_cache: Dict[int, Dict[str, xr.Dataset]] = {}
        self.mf_processor = MatrixFactorizationProcessor(config)

        logger.info(
            f"MLImputationProcessor (matrix factorization) initialized for {self.config.country}"
        )

    def process(self, missing_years: List[int]) -> List[Path]:
        """Main processing pipeline for matrix factorization imputation"""

        logger.info(
            f"Starting matrix factorization imputation for {len(missing_years)} missing years: {missing_years}"
        )

        # Step 1: Load and bound HYDE data to country
        self._load_hyde_data()

        # Step 2: Load available CCI data for training
        self._load_cci_data()

        # Step 3: Prepare data for matrix factorization
        hyde_array, cci_array = self._prepare_matrix_factorization_data()

        # Step 4: Train and predict using matrix factorization
        output_files = []
        output_dir = self.get_processed_subdirectory("cci_cropland_mask")

        for mask_type in ["cropland", "irrigated"]:
            logger.info(f"Processing {mask_type} mask with matrix factorization...")

            # Get mask-specific data
            cci_mask_array = self._extract_mask_data(cci_array, mask_type)

            # Train and predict
            predictions = self.mf_processor.train_and_predict(
                hyde_array, cci_mask_array, missing_years
            )

            # Save predictions
            mask_files = self._save_predictions(predictions, mask_type, output_dir)
            output_files.extend(mask_files)

        logger.info(
            f"Matrix factorization imputation completed. Generated {len(output_files)} files."
        )
        return output_files

    def _load_hyde_data(self) -> None:
        """Load HYDE data and bound to country geometry"""
        logger.info("Loading HYDE data...")

        # Get country bounds
        from src.utils.geography import Geography

        geography = Geography()
        bounds = geography.get_country_bounds(self.country_iso)

        # Load HYDE files
        hyde_dir = self.data_dir / "global" / "hyde-3.5"
        cropland_path = hyde_dir / "cropland.nc"

        if not cropland_path.exists():
            raise FileNotFoundError(f"HYDE cropland file not found: {cropland_path}")

        # Load full dataset
        hyde_cropland_full = xr.open_dataset(cropland_path)

        # Bound to country geometry with proper coordinate handling
        logger.info(
            f"Bounding HYDE data to {self.country_iso}: lat [{bounds.min_lat:.2f}, {bounds.max_lat:.2f}], lon [{bounds.min_lon:.2f}, {bounds.max_lon:.2f}]"
        )

        # Handle coordinate order (ascending vs descending)
        if hyde_cropland_full.lat.values[0] > hyde_cropland_full.lat.values[-1]:
            # Descending latitude order
            self.hyde_cropland = hyde_cropland_full.sel(
                lat=slice(bounds.max_lat, bounds.min_lat),
                lon=slice(bounds.min_lon, bounds.max_lon),
            )
        else:
            # Ascending latitude order
            self.hyde_cropland = hyde_cropland_full.sel(
                lat=slice(bounds.min_lat, bounds.max_lat),
                lon=slice(bounds.min_lon, bounds.max_lon),
            )

        # Validate bounded data
        if (
            self.hyde_cropland.sizes.get("lat", 0) == 0
            or self.hyde_cropland.sizes.get("lon", 0) == 0
        ):
            raise ValueError(
                f"HYDE data contains no pixels for {self.country_iso} bounds"
            )

        logger.info(
            f"HYDE data bounded - Shape: lat={self.hyde_cropland.sizes['lat']}, lon={self.hyde_cropland.sizes['lon']}"
        )

        # Close full dataset to save memory
        hyde_cropland_full.close()

    def _load_cci_data(self) -> None:
        """Load available CCI data for training"""
        logger.info("Loading CCI data for training...")

        cci_dir = self.get_processed_subdirectory("cci_cropland_mask")
        all_years = list(range(1992, 1997))  # CCI available years

        for year in tqdm(all_years, desc="Loading CCI data"):
            year_data = {}

            for mask_type in ["cropland", "irrigated"]:
                if mask_type == "cropland":
                    cci_file = cci_dir / f"cci_cropland_mask_{year}.nc"
                else:
                    cci_file = cci_dir / f"cci_cropland_mask_{mask_type}_{year}.nc"

                if cci_file.exists():
                    try:
                        year_data[mask_type] = xr.open_dataset(cci_file)
                        logger.debug(f"Loaded CCI {mask_type} data for {year}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load CCI data for {year}/{mask_type}: {e}"
                        )

            if year_data:
                self.cci_data_cache[year] = year_data

        logger.info(f"Loaded CCI data for {len(self.cci_data_cache)} years")

        if len(self.cci_data_cache) == 0:
            raise ValueError("No CCI data found for training")

    def _prepare_matrix_factorization_data(
        self,
    ) -> Tuple[xr.DataArray, Dict[str, xr.DataArray]]:
        """Prepare HYDE and CCI data for matrix factorization"""
        logger.info("Preparing data for matrix factorization...")

        if self.hyde_cropland is None:
            raise RuntimeError("HYDE data not loaded")

        # Use HYDE cropland as primary input
        hyde_array = self.hyde_cropland["cropland"]

        # Combine CCI data across years into time series
        cci_arrays = {}
        for mask_type in ["cropland", "irrigated"]:
            cci_time_series = []
            years = []

            for year in tqdm(
                sorted(self.cci_data_cache.keys()),
                desc=f"Preparing {mask_type} training data",
            ):
                if mask_type in self.cci_data_cache[year]:
                    cci_data = self.cci_data_cache[year][mask_type]

                    # Resample to match HYDE grid
                    cci_resampled = self._resample_to_hyde_grid(
                        cci_data["cci_cropland_mask"], self.hyde_cropland
                    )
                    cci_time_series.append(cci_resampled)
                    years.append(year)

            if cci_time_series:
                # Stack into time dimension
                cci_stacked = xr.concat(cci_time_series, dim="time")
                cci_stacked = cci_stacked.assign_coords(time=years)
                cci_arrays[mask_type] = cci_stacked

        logger.info(
            f"Prepared HYDE array: {hyde_array.shape}, CCI arrays: {list(cci_arrays.keys())}"
        )
        return hyde_array, cci_arrays

    def _resample_to_hyde_grid(
        self, cci_data: xr.DataArray, hyde_reference: xr.Dataset
    ) -> xr.DataArray:
        """Resample CCI data to match HYDE grid resolution"""
        resampled = cci_data.interp(
            lat=hyde_reference.lat,
            lon=hyde_reference.lon,
            method="linear",
        )
        return resampled

    def _extract_mask_data(
        self, cci_arrays: Dict[str, xr.DataArray], mask_type: str
    ) -> xr.DataArray:
        """Extract mask-specific CCI data"""
        if mask_type not in cci_arrays:
            raise ValueError(f"Mask type {mask_type} not found in CCI data")
        return cci_arrays[mask_type]

    def _save_predictions(
        self, predictions: Dict[int, xr.DataArray], mask_type: str, output_dir: Path
    ) -> List[Path]:
        """Save matrix factorization predictions to NetCDF files"""
        output_files = []

        for year, pred_array in predictions.items():
            # Create output dataset
            ds = xr.Dataset(
                {"cci_cropland_mask": pred_array},
                attrs={
                    "title": f"Matrix factorization CCI {mask_type} mask for {year}",
                    "source": "Matrix factorization imputation using HYDE data",
                    "model_type": "Coupled spatiotemporal matrix factorization",
                    "library": "MatCoupLy",
                    "spatial_rank": MatrixFactorizationConstants.SPATIAL_RANK,
                    "temporal_rank": MatrixFactorizationConstants.TEMPORAL_RANK,
                },
            )

            # Save file
            if mask_type == "cropland":
                filename = f"cci_cropland_mask_{year}.nc"
            else:
                filename = f"cci_cropland_mask_{mask_type}_{year}.nc"

            output_file = output_dir / filename
            ds.to_netcdf(output_file)
            output_files.append(output_file)

            logger.debug(
                f"Saved matrix factorization {mask_type} mask for {year}: {filename}"
            )

        return output_files
