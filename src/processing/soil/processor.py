"""Soil data processor - converts TIF files to NetCDF and aggregates to CSV"""

import logging
from pathlib import Path
from typing import List

import xarray as xr
import pandas as pd

from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.soil.config import SoilConfig
from src.processing.soil.formatter import SoilFormatter
from src.processing.soil.tiff_converter import SoilTiffConverter

logger = logging.getLogger(__name__)


class SoilProcessor(BaseProcessor):
    """Soil processor that converts TIF files to aggregated CSV"""

    def __init__(self, config: SoilConfig):
        super().__init__(
            config.country, config.admin_level, config.data_dir, config.debug
        )
        self.config = config
        self.spatial_aggregator = SpatialAggregator(
            self, config.country, cropland_filter=False
        )
        self.formatter = SoilFormatter()

        logger.info(f"SoilProcessor initialized for {config.country}")

    def process(self) -> List[Path]:
        """Process soil data from TIF to CSV"""
        logger.info(f"Processing soil data for {self.country_full_name}")

        soil_dir = self.config.get_soil_directory()
        intermediate_dir = self.get_intermediate_directory()
        soil_processed_dir = self.get_processed_subdirectory("soil")

        # Initialize soil TIF converter
        tiff_converter = SoilTiffConverter(soil_processed_dir)

        properties_to_process = self.config.properties
        str_props = "\n * ".join(properties_to_process)
        logger.info(f"Properties: \n * {str_props}")

        output_files = []

        for property_name in properties_to_process:
            logger.info(f"Processing property: {property_name}")

            # Step 1: Convert TIF files to NetCDF with depth dimension
            netcdf_file = tiff_converter.convert_property_to_netcdf(
                soil_dir, property_name, self.config.depths
            )

            # Step 2: Load the NetCDF file
            combined_ds = xr.open_dataset(netcdf_file)

            # Step 3: Process each depth separately since spatial aggregator expects time/week dimension
            all_depth_results = []

            for depth_key in combined_ds.depth.values:
                depth_ds = combined_ds.sel(depth=depth_key)

                # Add fake temporal structure for spatial aggregator compatibility
                depth_ds = self._add_fake_temporal_structure(depth_ds, year=2020)

                # Spatial aggregation
                aggregated_df = self.spatial_aggregator.aggregate_dataset(
                    depth_ds, f"{property_name}_{depth_key}"
                )

                # Add depth information
                aggregated_df["depth"] = depth_key
                aggregated_df["variable"] = property_name

                all_depth_results.append(aggregated_df)

            # Close the dataset
            combined_ds.close()

            # Combine all depths
            combined_df = pd.concat(all_depth_results, ignore_index=True)

            # Step 4: Format to pivot format
            pivoted_df = self.formatter.pivot_to_final_format(
                combined_df, self.config.admin_level
            )

            # Step 5: Save output
            filename = f"soil_{property_name}_weighted_admin{self.config.admin_level}.{self.config.output_format}"
            output_file = self.save_output(
                pivoted_df, filename, self.config.output_format, intermediate_dir
            )

            output_files.append(output_file)
            logger.debug(f"Completed processing for {property_name}: {output_file}")

        return output_files

    def _add_fake_temporal_structure(
        self, dataset: xr.Dataset, year: int = 2020
    ) -> xr.Dataset:
        """Add fake temporal structure to static data for spatial aggregator compatibility

        Args:
            dataset: Static dataset (e.g., soil data)
            year: Year to assign to the data (default: 2020 for soil data)

        Returns:
            Dataset with fake temporal structure (week dimension with year coordinate)
        """
        # Create fake temporal structure for spatial aggregator (soil data is from 2020)
        # Need to restructure to have week dimension like weather data
        dataset = dataset.expand_dims("time")
        dataset = dataset.assign_coords(time=[pd.Timestamp(f"{year}-01-01")])

        # Add year and week coordinates to time dimension
        dataset = dataset.assign_coords(year=("time", [year]))
        dataset = dataset.assign_coords(week=("time", [1]))

        # Restructure to have week dimension (like weather data after temporal aggregation)
        dataset = dataset.groupby("week").mean("time")
        dataset = dataset.assign_coords(year=("week", [year]))

        return dataset
