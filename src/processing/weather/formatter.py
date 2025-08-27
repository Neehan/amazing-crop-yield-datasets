"""Weather data formatter for converting weekly time series to pivot format"""

import logging

import pandas as pd

from src.processing.base.formatter import BaseFormatter
from src.constants import WEATHER_WEEKS_PER_YEAR

logger = logging.getLogger(__name__)


class WeatherFormatter(BaseFormatter):
    """Weather-specific formatter for weekly time series data"""

    def get_time_column_name(self) -> str:
        """Get the name of the time column for weather data"""
        return "time"

    def get_time_grouping_columns(self, data: pd.DataFrame) -> dict:
        """Get the columns to group by for time aggregation"""
        time_col = data[self.get_time_column_name()]

        # Use day-of-year based week calculation to match temporal aggregator
        day_of_year = time_col.dt.dayofyear
        week_number = ((day_of_year - 1) // 7) + 1

        return {"year": time_col.dt.year, "time_unit": week_number}

    def get_pivot_column_names(self, data: pd.DataFrame) -> dict:
        """Get mapping of week numbers to output column names"""
        variable_name = data.iloc[0]["variable"]
        # Use ordered generation to preserve week order
        column_mapping = {}
        for week in range(1, WEATHER_WEEKS_PER_YEAR + 1):
            column_mapping[week] = f"{variable_name}_week_{week}"
        return column_mapping
