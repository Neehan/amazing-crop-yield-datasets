"""Temporal aggregator for converting daily data to weekly averages"""

import logging
from pathlib import Path

import xarray as xr
import numpy as np
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
            weekly_data = self.daily_to_weekly_dataset(dataset)
            # Save as NetCDF
            weekly_data.to_netcdf(output_file)

        logger.debug(f"Weekly NetCDF saved: {output_file}")
        return output_file

    def daily_to_weekly_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Convert daily dataset to weekly averages in memory

        Args:
            dataset: Daily xarray Dataset

        Returns:
            Weekly averaged xarray Dataset
        """
        # Use day-of-year based weekly grouping instead of ISO calendar weeks
        # This ensures weeks start from day 1 and are consecutive 7-day periods
        day_of_year = dataset.time.dt.dayofyear
        week_number = ((day_of_year - 1) // 7) + 1  # Days 1-7→week 1, 8-14→week 2, etc.

        # Add week and year coordinates
        dataset = dataset.assign_coords(
            week=("time", week_number.data),
            year=("time", dataset.time.dt.year.data),
        )

        # Group by week within each year to preserve year information
        yearly_datasets = []
        unique_years = sorted(set(dataset.year.values))

        logger.debug(f"Computing weekly means for {len(unique_years)} years...")

        for year in tqdm(unique_years, desc="Processing years"):
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
