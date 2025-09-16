"""Crop calendar data processor"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm

from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.management.crop_calendar.config import (
    CropCalendarConfig,
    CROP_CODES,
)
from src.processing.management.crop_calendar.gz_converter import GZConverter
from src.processing.management.crop_calendar.ml_imputation import (
    CropCalendarMLImputation,
)

logger = logging.getLogger(__name__)


class CropCalendarProcessor(BaseProcessor):
    """Processes MIRCA2000 crop calendar data to administrative boundaries"""

    def __init__(self, config: CropCalendarConfig):
        """Initialize crop calendar processor

        Args:
            config: Crop calendar processing configuration
        """
        super().__init__(
            country=config.country,
            admin_level=config.admin_level,
            data_dir=config.data_dir,
            debug=config.debug,
        )
        self.config = config

        # Initialize components
        self.gz_converter = GZConverter(self)
        self.spatial_aggregator = SpatialAggregator(
            base_processor=self,
            country_param=config.country,
            cropland_filter=False,  # No cropland filtering for crop calendar data
            cache_dir_name="crop_calendar",
        )

        logger.info(
            f"CropCalendarProcessor initialized for {len(config.crop_codes)} crops"
        )

    def process(self) -> List[Path]:
        """Process crop calendar data for all configured crops

        Returns:
            List of paths to generated CSV files
        """
        output_files = []

        # Get unique crop names (not codes)
        unique_crop_names = list(
            set([CROP_CODES[code] for code in self.config.crop_codes])
        )

        for crop_name in unique_crop_names:
            logger.info(f"Processing crop: {crop_name} (weighted irrigated + rainfed)")

            # Convert GZ to NetCDF (now handles both irrigated and rainfed)
            netcdf_path = self.gz_converter.convert_crop_to_netcdf(crop_name)

            # Aggregate to administrative boundaries
            output_path = self._aggregate_to_admin_boundaries(netcdf_path, crop_name)

            if output_path:
                output_files.append(output_path)

        logger.info(f"Processing complete: {len(output_files)} files generated")
        return output_files

    def _aggregate_to_admin_boundaries(
        self, netcdf_path: Path, crop_name: str
    ) -> Optional[Path]:
        """Aggregate NetCDF crop calendar data to administrative boundaries

        Args:
            netcdf_path: Path to NetCDF file
            crop_name: Crop name

        Returns:
            Path to aggregated CSV file
        """
        # Check for cached result using base config
        final_dir = self.config.get_final_directory()
        final_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"crop_calendar_{crop_name}.csv"
        output_path = final_dir / "management" / output_filename

        if output_path.exists():
            logger.info(f"Using cached result: {output_path}")
            return output_path
        with xr.open_dataset(netcdf_path) as ds:
            if ds.area.sum() == 0:
                raise ValueError(
                    f"No data found for {crop_name} in {self.country_full_name}"
                )

        area_df = self._aggregate_spatial_data(netcdf_path, "area")
        if area_df.empty:
            raise ValueError(f"No area data after aggregation for {crop_name}")

        planted_dfs, harvested_dfs = self._process_monthly_data(netcdf_path)

        # Combine all data
        result_df = area_df[
            [
                "country_name",
                "admin_level_1_name",
                "admin_level_2_name",
                "admin_id",
                "latitude",
                "longitude",
            ]
        ].copy()
        result_df["total_area"] = area_df["value"]
        result_df["crop_name"] = crop_name

        # Add monthly columns
        for month_df in planted_dfs:
            result_df = result_df.merge(month_df, on="admin_id", how="left")

        for month_df in harvested_dfs:
            result_df = result_df.merge(month_df, on="admin_id", how="left")

        # Fill NaN values with 0 for monthly columns
        monthly_cols = [f"planted_month_{i}" for i in range(1, 13)] + [
            f"harvested_month_{i}" for i in range(1, 13)
        ]
        for col in monthly_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(0)  # type: ignore

        # Normalize monthly fractions so they sum to 1 for each admin unit
        planted_cols = [f"planted_month_{i}" for i in range(1, 13)]
        harvested_cols = [f"harvested_month_{i}" for i in range(1, 13)]

        # Calculate sums for normalization
        planted_sums = result_df[planted_cols].sum(axis=1)
        harvested_sums = result_df[harvested_cols].sum(axis=1)

        # Normalize only where sum > 0 (avoid division by zero)
        planted_mask = planted_sums > 0
        harvested_mask = harvested_sums > 0

        for col in planted_cols:
            if col in result_df.columns:
                result_df.loc[planted_mask, col] = (
                    result_df.loc[planted_mask, col] / planted_sums[planted_mask]
                )

        for col in harvested_cols:
            if col in result_df.columns:
                result_df.loc[harvested_mask, col] = (
                    result_df.loc[harvested_mask, col] / harvested_sums[harvested_mask]
                )

        # Remove admin_id column from final output
        result_df = result_df.drop(columns=["admin_id"])
        result_df = pd.DataFrame(result_df)

        # Apply ML imputation for missing data
        result_df = self._apply_ml_imputation(result_df, crop_name)

        # Save result
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved crop calendar data: {output_path}")

        return output_path

    def _aggregate_spatial_data(
        self, netcdf_path: Path, variable_name: str
    ) -> pd.DataFrame:
        """Custom spatial aggregation for crop calendar data (no time dimension)"""

        with xr.open_dataset(netcdf_path) as ds:
            # Get the data variable
            data_var = ds[variable_name]

            # Create admin mask if not exists
            lats = data_var.lat.values
            lons = data_var.lon.values
            lat_step = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.1
            lon_step = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.1

            admin_mask = self.spatial_aggregator._load_or_create_admin_mask(
                lats, lons, lat_step, lon_step
            )

            # Simple aggregation without time dimension
            results = []
            data_values = data_var.values

            # Get valid admin units
            valid_admin_ids = []
            for admin_id in range(len(self.boundaries)):
                admin_cells = admin_mask == admin_id
                if admin_cells.any() and not np.isnan(data_values[admin_cells]).all():
                    area_weights = self.spatial_aggregator.area_weights[:, :, admin_id]
                    weighted_sum = np.sum(data_values * area_weights * admin_cells)
                    total_weight = np.sum(area_weights * admin_cells)

                    if total_weight > 0:
                        avg_value = weighted_sum / total_weight
                        valid_admin_ids.append((admin_id, avg_value))

            # Convert to DataFrame format matching spatial aggregator output
            for admin_id, avg_value in valid_admin_ids:
                admin_row = self.boundaries.iloc[admin_id]
                centroid = self.spatial_aggregator.boundary_centroids.iloc[admin_id]

                result_dict = {
                    "admin_name": self.get_admin_name(admin_row),
                    "admin_id": admin_id,
                    "value": avg_value,
                    "country_name": self.country_full_name,
                    "latitude": round(centroid.y, 3),
                    "longitude": round(centroid.x, 3),
                }

                # Add admin level names
                for level in range(1, self.admin_level + 1):
                    result_dict[f"admin_level_{level}_name"] = admin_row.get(
                        f"NAME_{level}", ""
                    )

                results.append(result_dict)

            return pd.DataFrame(results)

    def _process_monthly_data(self, netcdf_path: Path) -> tuple[list, list]:
        """Process monthly planted and harvested data from NetCDF"""

        planted_dfs = []
        harvested_dfs = []

        with xr.open_dataset(netcdf_path) as ds:
            for month in tqdm(range(1, 13), desc="Processing monthly data"):
                # Process planted data
                planted_df = self._process_monthly_variable(
                    ds, "planted", month, netcdf_path.parent
                )
                if planted_df is not None:
                    planted_dfs.append(planted_df)

                # Process harvested data
                harvested_df = self._process_monthly_variable(
                    ds, "harvested", month, netcdf_path.parent
                )
                if harvested_df is not None:
                    harvested_dfs.append(harvested_df)

        return planted_dfs, harvested_dfs

    def _process_monthly_variable(
        self, ds: xr.Dataset, variable_name: str, month: int, temp_dir: Path
    ) -> Optional[pd.DataFrame]:
        """Process a single monthly variable (planted or harvested)

        Args:
            ds: xarray Dataset containing the data
            variable_name: Name of the variable to process ('planted' or 'harvested')
            month: Month number (1-12)
            temp_dir: Directory for temporary files

        Returns:
            DataFrame with admin_id and monthly column, or None if empty
        """
        month_data = ds[variable_name].isel(month=month - 1)
        temp_ds = xr.Dataset({variable_name: month_data})
        temp_path = temp_dir / f"temp_{variable_name}_month_{month}.nc"
        temp_ds.to_netcdf(temp_path)

        df = self._aggregate_spatial_data(temp_path, variable_name)
        temp_path.unlink()

        if not df.empty:
            df[f"{variable_name}_month_{month}"] = df["value"]
            return pd.DataFrame(df[["admin_id", f"{variable_name}_month_{month}"]])

        raise ValueError(f"No data found for {variable_name} in month {month}")

    def _get_admin_units_from_df(self, df: pd.DataFrame, col1: str, col2: str) -> set:
        """Extract unique admin units from DataFrame columns"""
        return set(df[[col1, col2]].apply(lambda x: (x[col1], x[col2]), axis=1))

    def _filter_by_yield_data(
        self, crop_calendar_df: pd.DataFrame, crop_name: str
    ) -> pd.DataFrame:
        """Filter admin units to only those that have yield data for the specific crop"""
        yield_file = (
            self.config.get_final_directory() / "crop" / f"crop_{crop_name}_yield.csv"
        )
        yield_df = pd.read_csv(yield_file)

        yield_admin_units = self._get_admin_units_from_df(
            yield_df, "admin_level_1", "admin_level_2"
        )
        crop_calendar_admin_units = self._get_admin_units_from_df(
            crop_calendar_df, "admin_level_1_name", "admin_level_2_name"
        )

        common_admin_units = yield_admin_units.intersection(crop_calendar_admin_units)
        mask = crop_calendar_df[["admin_level_1_name", "admin_level_2_name"]].apply(
            lambda x: (x["admin_level_1_name"], x["admin_level_2_name"])
            in common_admin_units,
            axis=1,
        )

        filtered_df = pd.DataFrame(crop_calendar_df[mask].copy())
        logger.info(
            f"Yield filtering for {crop_name}: {len(crop_calendar_df)} -> {len(filtered_df)} admin units"
        )
        return filtered_df

    def _apply_ml_imputation(
        self, crop_calendar_df: pd.DataFrame, crop_name: str
    ) -> pd.DataFrame:
        """Apply ML imputation to fill missing crop calendar data"""
        filtered_df = self._filter_by_yield_data(crop_calendar_df, crop_name)

        ml_imputation = CropCalendarMLImputation(
            country=self.country_full_name,
            admin_level=self.admin_level,
            data_dir=self.config.data_dir,
        )

        temp_path = (
            self.config.get_processed_subdirectory("crop_calendar")
            / f"temp_crop_calendar_{crop_name}.csv"
        )
        filtered_df.to_csv(temp_path, index=False)

        imputed_df = ml_imputation.impute_crop_calendar(temp_path, year=2000)
        temp_path.unlink()

        zero_area_records = (filtered_df["total_area"] == 0.0).sum()
        monthly_cols = [col for col in imputed_df.columns if "month_" in col]
        imputed_records = (
            (imputed_df.loc[imputed_df["total_area"] == 0.0, monthly_cols] > 0)
            .any(axis=1)
            .sum()
        )

        logger.info(
            f"ML imputation for {crop_name}: raw: {len(crop_calendar_df)} -> filtered by yield: {len(filtered_df)} -> imputed: {len(imputed_df)} "
            f"({zero_area_records} zero area, {imputed_records} imputed)"
        )
        return imputed_df
