"""Temporal aggregator for converting daily data to weekly averages"""

import logging
from pathlib import Path

import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """Aggregates temporal data (daily to weekly) for any data type"""

    def __init__(self):
        """Initialize temporal aggregator"""
        logger.info("Temporal aggregator initialized")

    def daily_to_weekly_netcdf(self, input_file: str, output_file: str) -> str:
        """Convert daily NetCDF data to weekly averages and save as NetCDF

        Args:
            input_file: Path to daily NetCDF file
            output_file: Path to save weekly NetCDF file

        Returns:
            Path to weekly NetCDF file
        """
        logger.debug(f"Converting daily to weekly: {input_file} -> {output_file}")

        with xr.open_dataset(input_file) as dataset:
            # Add week and year coordinates
            dataset = dataset.assign_coords(
                week=("time", dataset.time.dt.isocalendar().week.data),
                year=("time", dataset.time.dt.year.data),
            )

            # Group by year and week, take mean
            # Show progress by wrapping the groupby operation
            groups = dataset.groupby("week")
            logger.debug(f"Computing weekly means for {len(groups)} weeks...")
            
            # Use tqdm to show progress on the mean calculation
            with tqdm(total=1, desc="Computing weekly means") as pbar:
                weekly_data = groups.mean("time")
                pbar.update(1)
            
            # Add year coordinate back - use middle time point to avoid edge cases
            middle_idx = len(dataset.time) // 2
            year_for_all_weeks = dataset.time.dt.year.values[middle_idx]
            weekly_data = weekly_data.assign_coords(year=("week", [year_for_all_weeks] * len(weekly_data.week)))

            # Save as NetCDF
            weekly_data.to_netcdf(output_file)

        logger.debug(f"Weekly NetCDF saved: {output_file}")
        return output_file
