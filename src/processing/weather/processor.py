"""Weather data processor that inherits from base processor"""

import logging
from pathlib import Path
from typing import List

from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.base.temporal_aggregator import TemporalAggregator
from src.processing.weather.config import WeatherConfig
from src.processing.weather.formatter import WeatherFormatter
from src.processing.base.zip_extractor import ZipExtractor

logger = logging.getLogger(__name__)


class WeatherProcessor(BaseProcessor):
    """Weather-specific processor that inherits base functionality"""

    def __init__(self, config: WeatherConfig):
        """Initialize weather processor with configuration"""
        super().__init__(config.country, config.admin_level, config.data_dir)
        self.config = config
        self.spatial_aggregator = SpatialAggregator(self, config.country, cropland_filter=True)
        self.temporal_aggregator = TemporalAggregator()
        self.formatter = WeatherFormatter()

        # Set up logging
        log_level = logging.DEBUG if config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info(f"WeatherProcessor initialized for {config.country}")

    def process(self) -> List[Path]:
        """Process weather data according to configuration"""
        logger.info(
            f"Processing weather data for {self.country_full_name} ({self.config.start_year}-{self.config.end_year})"
        )

        # Validate configuration
        self.config.validate()

        # Get weather directory and processed output directory
        weather_dir = self.config.get_weather_directory()
        processed_dir = self.data_dir / self.country_full_name.lower() / "processed"
        weather_processed_dir = processed_dir / "weather"
        weather_processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize zip extractor with weather processed directory
        zip_extractor = ZipExtractor(weather_processed_dir)

        # Get available variables if not specified
        variables_to_process = self.config.variables
        if not variables_to_process:
            available_data = zip_extractor.get_available_data(weather_dir)
            variables_to_process = sorted(list(set(var for _, var in available_data)))
            str_vars = "\n * ".join(variables_to_process)
            logger.info(f"Auto-detected variables: \n * {str_vars}")

        output_files = []

        # Process each variable separately
        for variable in variables_to_process:
            logger.info(f"Processing variable: {variable}")

            # Step 1: Extract zip files for this variable into annual NetCDF files
            logger.debug(f"Extracting zip files for {variable}...")
            daily_nc_files = zip_extractor.process_all_zips(
                weather_dir=weather_dir,
                year_range=(self.config.start_year, self.config.end_year),
                variables=[variable],
                force_refresh=False,
            )

            # Step 2: Combine annual files into single daily NetCDF (in memory, don't save)
            logger.debug(f"Combining annual files for {variable}...")
            combined_ds = self.combine_annual_files_in_memory(daily_nc_files)

            # Step 3: Temporal aggregation - convert daily to weekly (in memory, don't save)
            logger.debug(f"Converting {variable} from daily to weekly...")
            weekly_ds = self.temporal_aggregator.daily_to_weekly_dataset(combined_ds)

            # Step 4: Spatial aggregation - aggregate to admin boundaries
            logger.debug(f"Spatially aggregating {variable} to admin boundaries...")
            aggregated_df = self.spatial_aggregator.aggregate_dataset(weekly_ds)

            # Step 5: Format and save output
            logger.debug(f"Formatting and saving output for {variable}...")

            # Add variable column for formatting
            aggregated_df["variable"] = variable

            # Convert to pivot format
            pivoted_df = self.formatter.pivot_to_final_format(
                aggregated_df, self.config.admin_level
            )

            # Save output file
            filename = f"weather_{self.config.start_year}-{self.config.end_year}_{variable}_weekly_weighted_admin{self.config.admin_level}.{self.config.output_format}"
            output_file = self.save_output(
                pivoted_df, filename, self.config.output_format, processed_dir
            )

            output_files.append(output_file)
            logger.debug(f"Completed processing for {variable}: {output_file}")

        logger.info(f"Weather processing complete. Output files: {output_files}")
        return output_files
