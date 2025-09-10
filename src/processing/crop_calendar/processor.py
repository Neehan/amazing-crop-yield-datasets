"""Crop calendar data processor"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.crop_calendar.config import CropCalendarConfig, CROP_CODES
from src.processing.crop_calendar.gz_converter import GZConverter
from src.processing.crop_calendar.ml_imputation import CropCalendarMLImputation

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
        # Check for cached result
        final_dir = Path("data") / self.country_full_name.lower() / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"crop_calendar_{crop_name}.csv"
        output_path = final_dir / output_filename

        if output_path.exists():
            logger.info(f"Using cached result: {output_path}")
            return output_path

        # Load NetCDF and check if it has data
        import xarray as xr

        with xr.open_dataset(netcdf_path) as ds:
            # Check if dataset has any non-zero area
            if ds.area.sum() == 0:
                logger.warning(
                    f"No data found for {crop_name} in {self.country_full_name}"
                )
                return None

        # Use custom spatial aggregation for crop calendar data
        area_df = self._aggregate_spatial_data(netcdf_path, "area")

        if area_df.empty:
            logger.warning(f"No area data after aggregation for {crop_name}")
            return None

        # Aggregate monthly planted and harvested data directly from main NetCDF
        planted_dfs = []
        harvested_dfs = []

        import xarray as xr

        with xr.open_dataset(netcdf_path) as ds:
            for month in range(1, 13):
                # Planted data for this month
                month_planted = ds.planted.isel(month=month - 1)
                temp_planted_ds = xr.Dataset({"planted": month_planted})
                temp_planted_path = (
                    netcdf_path.parent / f"temp_planted_month_{month}.nc"
                )
                temp_planted_ds.to_netcdf(temp_planted_path)

                planted_df = self._aggregate_spatial_data(temp_planted_path, "planted")
                if not planted_df.empty:
                    planted_df[f"planted_month_{month}"] = planted_df["value"]
                    planted_dfs.append(
                        planted_df[["admin_id", f"planted_month_{month}"]]
                    )

                temp_planted_path.unlink()

                # Harvested data for this month
                month_harvested = ds.harvested.isel(month=month - 1)
                temp_harvested_ds = xr.Dataset({"harvested": month_harvested})
                temp_harvested_path = (
                    netcdf_path.parent / f"temp_harvested_month_{month}.nc"
                )
                temp_harvested_ds.to_netcdf(temp_harvested_path)

                harvested_df = self._aggregate_spatial_data(
                    temp_harvested_path, "harvested"
                )
                if not harvested_df.empty:
                    harvested_df[f"harvested_month_{month}"] = harvested_df["value"]
                    harvested_dfs.append(
                        harvested_df[["admin_id", f"harvested_month_{month}"]]
                    )

                temp_harvested_path.unlink()

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
        import xarray as xr
        import numpy as np

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

    def _apply_ml_imputation(
        self, crop_calendar_df: pd.DataFrame, crop_name: str
    ) -> pd.DataFrame:
        """Apply ML imputation to fill missing crop calendar data

        Args:
            crop_calendar_df: Crop calendar DataFrame
            crop_name: Name of the crop

        Returns:
            DataFrame with imputed data
        """
        # Initialize ML imputation
        ml_imputation = CropCalendarMLImputation(
            country=self.country_full_name,
            admin_level=self.admin_level,
            data_dir=Path("data"),
        )

        # Create temporary file for ML imputation
        temp_path = (
            Path("data")
            / self.country_full_name.lower()
            / "intermediate"
            / "crop_calendar"
            / f"temp_crop_calendar_{crop_name}.csv"
        )
        crop_calendar_df.to_csv(temp_path, index=False)

        # Apply ML imputation (use 2000 to match MIRCA2000 crop calendar data)
        imputed_df = ml_imputation.impute_crop_calendar(temp_path, year=2000)

        # Clean up temporary file
        temp_path.unlink()

        # Report statistics
        original_count = len(crop_calendar_df)
        imputed_count = len(imputed_df)
        new_records = imputed_count - original_count

        logger.info(f"ML imputation results for {crop_name}:")
        logger.info(f"  Original records: {original_count}")
        logger.info(f"  New imputed records: {new_records}")
        logger.info(f"  Total records: {imputed_count}")
        logger.info(f"  Coverage improvement: {new_records/original_count*100:.1f}%")

        return imputed_df
