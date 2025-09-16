"""Irrigation fraction processor - simple approach"""

import logging
from pathlib import Path
from typing import List
import pandas as pd
import xarray as xr

from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.management.irrigation.config import IrrigationConfig
from src.processing.management.irrigation.formatter import IrrigationFormatter
from src.utils.geography import Geography

logger = logging.getLogger(__name__)


class IrrigationProcessor(BaseProcessor):
    """Compute irrigation fraction = irrigation_area_fraction / cropland_area_fraction"""

    def __init__(self, config: IrrigationConfig):
        super().__init__(
            config.country, config.admin_level, config.data_dir, config.debug
        )
        self.config = config
        self.geography = Geography()
        self.formatter = IrrigationFormatter()

        # No cropland filtering - we want total areas
        self.spatial_aggregator = SpatialAggregator(
            self, config.country, cropland_filter=True, cache_dir_name="irrigation"
        )

    def process(self) -> List[Path]:
        """Process irrigation fraction"""
        logger.info(f"Computing irrigation fraction for {self.country_full_name}")

        # Spatial aggregation automatically computes fraction of total area
        irrigation_df = self._compute_area_fraction()

        # Rename value column to irrigated_fraction
        irrigation_df["irrigated_fraction"] = irrigation_df["value"]
        irrigation_df = irrigation_df.drop(columns=["value"])

        # Format the data with required columns and types
        formatted_df = self.formatter.format_irrigation_data(
            irrigation_df, self.country_full_name, self.config.admin_level
        )

        # Save result
        output_path = (
            self.config.get_final_directory()
            / f"irrigated_fraction_admin{self.config.admin_level}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        formatted_df = formatted_df.sort_values(
            ["country", "admin_level_1", "admin_level_2", "year"]
        )
        formatted_df.to_csv(output_path, index=False)
        logger.info(f"Saved irrigation fraction: {output_path}")

        return [output_path]

    def _filter_dataset_by_country_bounds(self, dataset: xr.Dataset) -> xr.Dataset:
        """Filter dataset by country bounding box to improve performance"""
        country_bounds = self.geography.get_country_bounds(
            self.country_full_name, buffer_degrees=0.5
        )
        logger.info(
            f"Filtering data to country bounds: lat={country_bounds.min_lat:.2f} to {country_bounds.max_lat:.2f}, "
            f"lon={country_bounds.min_lon:.2f} to {country_bounds.max_lon:.2f}"
        )

        return dataset.sel(
            lat=slice(country_bounds.max_lat, country_bounds.min_lat),
            lon=slice(country_bounds.min_lon, country_bounds.max_lon),
        )

    def _compute_area_fraction(self) -> pd.DataFrame:
        """Compute irrigation area fraction = total_irrigated / total_cropland"""
        # Load both irrigation and cropland data
        irrigation_ds = xr.open_dataset(self.config.get_irrigation_file())
        cropland_ds = xr.open_dataset(self.config.get_cropland_file())

        # Filter both datasets by country bounds
        irrigation_filtered = self._filter_dataset_by_country_bounds(irrigation_ds)
        cropland_filtered = self._filter_dataset_by_country_bounds(cropland_ds)

        # Filter years
        year_slice = slice(f"{self.config.start_year}", f"{self.config.end_year-1}")
        irrigation_data = irrigation_filtered.sel(time=year_slice)["total_irrigated"]
        cropland_data = cropland_filtered.sel(time=year_slice)["cropland"]

        # Compute irrigation fraction = irrigated_area / cropland_area
        # Avoid division by zero
        irrigated_fraction = xr.where(
            cropland_data > 0, irrigation_data / cropland_data, 0.0
        )

        # Irrigation is annual - treat each year as a separate "week"
        years = [int(t) for t in irrigation_data.time.dt.year.values]
        weeks = [1] * len(years)

        # Create new data array with week = 1 for all years
        data_array = xr.DataArray(
            irrigated_fraction.values,
            dims=["week", "lat", "lon"],
            coords={
                "week": weeks,
                "lat": irrigation_data.lat,
                "lon": irrigation_data.lon,
                "year": ("week", years),
            },
        )
        dataset = xr.Dataset({"irrigated_fraction": data_array})

        # Aggregate - returns irrigation fraction
        return self.spatial_aggregator.aggregate_dataset(dataset, "irrigated_fraction")
