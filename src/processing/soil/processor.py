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
        super().__init__(config.country, config.admin_level, config.data_dir)
        self.config = config
        self.spatial_aggregator = SpatialAggregator(
            self, config.country, cropland_filter=False
        )
        self.formatter = SoilFormatter()

        log_level = logging.DEBUG if config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info(f"SoilProcessor initialized for {config.country}")

    def process(self) -> List[Path]:
        """Process soil data from TIF to CSV"""
        logger.info(f"Processing soil data for {self.country_full_name}")

        self.config.validate()

        soil_dir = self.config.get_soil_directory()
        processed_dir = self.data_dir / self.country_full_name.lower() / "processed"
        soil_processed_dir = processed_dir / "soil"
        soil_processed_dir.mkdir(parents=True, exist_ok=True)

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

                # Create a fake time dimension for spatial aggregator
                depth_ds = depth_ds.expand_dims("time")
                depth_ds = depth_ds.assign_coords(time=[pd.Timestamp("2020-01-01")])

                # Spatial aggregation
                aggregated_df = self.spatial_aggregator.aggregate_dataset(depth_ds)

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
                pivoted_df, filename, self.config.output_format, processed_dir
            )

            output_files.append(output_file)
            logger.debug(f"Completed processing for {property_name}: {output_file}")

        logger.info(f"Soil processing complete. Output files: {output_files}")
        return output_files
