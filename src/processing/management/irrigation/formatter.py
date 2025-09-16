"""Formatter for irrigation data"""

import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class IrrigationFormatter:
    """Formats irrigation data to standard output format"""

    def __init__(self):
        """Initialize irrigation formatter"""
        pass

    def format_irrigation_data(
        self, df: pd.DataFrame, country: str, admin_level: int
    ) -> pd.DataFrame:
        """Format irrigation data with required columns and data types

        Args:
            df: DataFrame with irrigation data from spatial aggregator
            country: Country name
            admin_level: Administrative level

        Returns:
            Formatted DataFrame with columns: country, admin_level_1, admin_level_2,
            latitude, longitude, year, irrigated_fraction
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to formatter")
            empty_df = pd.DataFrame()
            empty_df["country"] = []
            empty_df["admin_level_1"] = []
            empty_df["admin_level_2"] = []
            empty_df["latitude"] = []
            empty_df["longitude"] = []
            empty_df["year"] = []
            empty_df["irrigated_fraction"] = []
            return empty_df

        # Create formatted DataFrame with proper structure
        formatted_df = pd.DataFrame(
            {
                "country": [country] * len(df),
                "admin_level_1": (
                    df["admin_level_1_name"] if admin_level >= 1 else [""] * len(df)
                ),
                "admin_level_2": (
                    df["admin_level_2_name"] if admin_level >= 2 else [""] * len(df)
                ),
                "latitude": df["latitude"].round(3),
                "longitude": df["longitude"].round(3),
                "year": pd.to_datetime(df["time"]).dt.year,
                "irrigated_fraction": df["irrigated_fraction"],
            }
        )

        logger.info(f"Formatted irrigation data: {len(formatted_df)} records")
        return formatted_df
