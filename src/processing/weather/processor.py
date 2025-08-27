"""Weather data processor that inherits from base processor"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import xarray as xr
from tqdm import tqdm

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
        self.spatial_aggregator = SpatialAggregator(self, config.country)
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
            variables_to_process = list(set(var for _, var in available_data))
            logger.debug(f"Auto-detected variables: {variables_to_process}")

        output_files = []

        # Process each variable separately
        for variable in variables_to_process:
            logger.info(f"Processing variable: {variable}")

            # Step 1: Extract and combine zip files for this variable into annual NetCDF files
            logger.debug(f"Extracting and combining zip files for {variable}...")
            annual_nc_files = zip_extractor.process_all_zips(
                weather_dir=weather_dir,
                year_range=(self.config.start_year, self.config.end_year),
                variables=[variable],
                force_refresh=False,
            )

            # Step 2: Combine annual files into single daily NetCDF (in memory, don't save)
            logger.debug(f"Combining annual files for {variable}...")
            combined_ds = self._combine_annual_files_in_memory(annual_nc_files)

            # Step 3: Temporal aggregation - convert daily to weekly (in memory, don't save)
            logger.debug(f"Converting {variable} from daily to weekly...")
            weekly_ds = self._daily_to_weekly_in_memory(combined_ds)

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
            output_file = processed_dir / filename

            if self.config.output_format == "csv":
                pivoted_df.to_csv(output_file, index=False)
            elif self.config.output_format == "parquet":
                pivoted_df.to_parquet(output_file, index=False)
            else:
                raise ValueError(
                    f"Unsupported output format: {self.config.output_format}"
                )

            output_files.append(output_file)
            logger.debug(f"Completed processing for {variable}: {output_file}")

        logger.info(f"Weather processing complete. Output files: {output_files}")
        return output_files

    def _combine_annual_files_in_memory(self, annual_files: List[Path]) -> xr.Dataset:
        """Combine annual NetCDF files in memory without saving"""
        datasets = []
        for file_path in sorted(annual_files):
            ds = xr.open_dataset(file_path)
            datasets.append(ds)

        # Concatenate along time dimension
        combined_ds = xr.concat(datasets, dim="time")
        combined_ds = combined_ds.sortby("time")
        return combined_ds

    def _daily_to_weekly_in_memory(self, dataset: xr.Dataset) -> xr.Dataset:
        """Convert daily data to weekly averages in memory without saving"""
        # Add week and year coordinates
        dataset = dataset.assign_coords(
            week=("time", dataset.time.dt.isocalendar().week.data),
            year=("time", dataset.time.dt.year.data),
        )

        # Group by week within each year to preserve year information
        # First group by year, then by week within each year
        yearly_datasets = []
        unique_years = sorted(set(dataset.year.values))

        for year in unique_years:
            year_data = dataset.where(dataset.year == year, drop=True)
            if len(year_data.time) > 0:
                # Group by week for this year only
                year_weekly = year_data.groupby("week").mean("time")
                # Add year coordinate
                year_weekly = year_weekly.assign_coords(
                    year=("week", [year] * len(year_weekly.week))
                )
                yearly_datasets.append(year_weekly)

        # Combine all years back together
        if yearly_datasets:
            weekly_data = xr.concat(yearly_datasets, dim="week")
        else:
            # Fallback to original method if no data
            groups = dataset.groupby("week")
            weekly_data = groups.mean("time")
            middle_idx = len(dataset.time) // 2
            year_for_all_weeks = dataset.time.dt.year.values[middle_idx]
            weekly_data = weekly_data.assign_coords(
                year=("week", [year_for_all_weeks] * len(weekly_data.week))
            )

        return weekly_data

    def _combine_annual_files(
        self, annual_files: List[Path], variable: str, output_dir: Path
    ) -> Path:
        """Combine multiple annual NetCDF files into single daily file

        Args:
            annual_files: List of annual NetCDF file paths
            variable: Variable name
            output_dir: Output directory

        Returns:
            Path to combined daily NetCDF file
        """
        import xarray as xr

        output_file = (
            output_dir
            / f"{self.country_full_name.lower()}_{variable}_daily_{self.config.start_year}-{self.config.end_year}.nc"
        )

        if output_file.exists():
            logger.debug(f"Using existing combined daily file: {output_file}")
            return output_file

        logger.debug(f"Combining {len(annual_files)} annual files into {output_file}")

        # Load all annual datasets
        datasets = []
        for file_path in sorted(annual_files):
            ds = xr.open_dataset(file_path)
            datasets.append(ds)

        # Concatenate along time dimension
        combined_ds = xr.concat(datasets, dim="time")
        combined_ds = combined_ds.sortby("time")

        # Add metadata
        combined_ds.attrs.update(
            {
                "title": f"Combined daily {variable} data for {self.config.start_year}-{self.config.end_year}",
                "variable": variable,
                "country": self.country_full_name,
                "processing_date": pd.Timestamp.now().isoformat(),
            }
        )

        # Save combined dataset
        combined_ds.to_netcdf(output_file)

        # Close datasets
        for ds in datasets:
            ds.close()
        combined_ds.close()

        logger.debug(f"Successfully combined into {output_file}")
        return output_file
