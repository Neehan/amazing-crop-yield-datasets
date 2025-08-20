"""Temporal aggregator for converting daily data to weekly averages"""

import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class TemporalAggregator:
    """Aggregates temporal data (daily to weekly, monthly, etc.)"""

    def __init__(self):
        """Initialize temporal aggregator"""
        logger.info("Temporal aggregator initialized")

    def daily_to_weekly_pivot(
        self, data: pd.DataFrame, admin_level: int = 2, time_column: str = "time"
    ) -> pd.DataFrame:
        """Convert daily data to weekly averages and pivot to final format

        Args:
            data: DataFrame with daily time series data (time, admin_name, admin_id, value, country_name, admin_level_1_name, admin_level_2_name, variable)
            admin_level: GADM admin level being processed (determines which admin columns to include)
            time_column: Name of time column

        Returns:
            DataFrame with columns: country, admin_level_1, admin_level_2, year, {variable}_week_1, {variable}_week_2, ..., {variable}_week_52
        """
        logger.debug("Converting daily data to weekly pivot format")

        assert (
            time_column in data.columns
        ), f"Time column '{time_column}' not found in data"

        # Ensure time column is datetime
        df = data.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        # Extract year and week number
        df["year"] = df[time_column].dt.year
        df["week_num"] = df[time_column].dt.isocalendar().week

        # Drop week 53 data (leap year edge cases) to ensure exactly 52 weeks per year
        df = df[df["week_num"] <= 52].copy()

        # Get the value column and variable name
        admin_cols = [
            "country_name",
            "admin_level_1_name",
            "admin_level_2_name",
            "admin_id",
            "time",
            "year",
            "week_num",
            "variable",
        ]
        value_cols = [col for col in df.columns if col not in admin_cols]
        assert (
            len(value_cols) == 1
        ), f"Expected 1 value column, got {len(value_cols)}: {value_cols}"
        value_col = value_cols[0]

        # Get variable name for column naming
        assert "variable" in df.columns, "Variable column required for proper naming"
        variable_name = df.iloc[0][
            "variable"
        ]  # Should be same for all rows in single processing run

        # Build grouping columns based on admin level
        grouping_cols = ["country_name", "year", "week_num"]
        if admin_level >= 1:
            grouping_cols.append("admin_level_1_name")
        if admin_level >= 2:
            grouping_cols.append("admin_level_2_name")

        # Group by admin, year, week and take mean
        weekly_agg = df.groupby(grouping_cols)[value_col].mean().reset_index()

        # Build index columns for pivot
        index_cols = ["country_name", "year"]
        if admin_level >= 1:
            index_cols.append("admin_level_1_name")
        if admin_level >= 2:
            index_cols.append("admin_level_2_name")

        # Pivot: rows = admin hierarchy + year, columns = week_num, values = weather value
        pivoted_table = weekly_agg.pivot_table(
            index=index_cols, columns="week_num", values=value_col, aggfunc="mean"
        )
        pivoted = pd.DataFrame(pivoted_table).reset_index()

        # Rename week columns to {variable}_week_1, {variable}_week_2, etc.
        week_cols = {
            col: f"{variable_name}_week_{col}"
            for col in pivoted.columns
            if isinstance(col, (int, float))
        }
        pivoted = pivoted.rename(columns=week_cols)

        # Fill missing weeks with NaN (some years might not have all 52 weeks)
        for week in range(1, 53):
            week_col = f"{variable_name}_week_{week}"
            if week_col not in pivoted.columns:
                pivoted[week_col] = np.nan

        # Sort columns properly - admin hierarchy + year + weeks
        final_cols = ["country_name"]
        if admin_level >= 1:
            final_cols.append("admin_level_1_name")
        if admin_level >= 2:
            final_cols.append("admin_level_2_name")
        final_cols.append("year")
        final_cols.extend([f"{variable_name}_week_{i}" for i in range(1, 53)])

        # Rename columns to final format
        column_mapping = {
            "country_name": "country",
            "admin_level_1_name": "admin_level_1",
            "admin_level_2_name": "admin_level_2",
        }
        pivoted = pivoted.rename(columns=column_mapping)

        # Select final columns (only those that exist)
        final_cols = [column_mapping.get(col, col) for col in final_cols]
        existing_cols = [col for col in final_cols if col in pivoted.columns]
        pivoted = pivoted[existing_cols]

        # Reset index for clean output
        pivoted = pivoted.reset_index(drop=True)

        logger.debug(f"Weekly pivot complete: {pivoted.shape}")
        return pd.DataFrame(pivoted)

    def daily_to_monthly(
        self, data: pd.DataFrame, time_column: str = "time"
    ) -> pd.DataFrame:
        """Convert daily data to monthly averages"""
        logger.info("Converting daily data to monthly averages")

        assert (
            time_column in data.columns
        ), f"Time column '{time_column}' not found in data"

        # Ensure time column is datetime
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])

        # Create month period
        data["month"] = data[time_column].dt.to_period("M")

        # Group by admin units and month
        grouping_cols = ["admin_name", "admin_id", "month"]
        if "variable" in data.columns:
            grouping_cols.append("variable")

        # Get numeric columns for aggregation
        numeric_cols = data.select_dtypes(include=["number"]).columns
        numeric_cols = [col for col in numeric_cols if col not in ["admin_id"]]

        # Aggregate by month
        monthly_data = data.groupby(grouping_cols)[numeric_cols].mean().reset_index()

        # Convert month period back to timestamp
        monthly_data["time"] = monthly_data["month"].dt.start_time
        monthly_data = monthly_data.drop("month", axis=1)

        logger.info(f"Monthly aggregation complete: {len(monthly_data)} records")
        return monthly_data

    def aggregate_temporal(
        self,
        data: pd.DataFrame,
        admin_level: int = 2,
        aggregation: str = "weekly",
        time_column: str = "time",
    ) -> pd.DataFrame:
        """Generic temporal aggregation method

        Args:
            data: DataFrame with time series data
            admin_level: GADM admin level being processed
            aggregation: Aggregation period (weekly, monthly, daily)
            time_column: Name of time column

        Returns:
            Temporally aggregated DataFrame
        """
        if aggregation == "weekly":
            return self.daily_to_weekly_pivot(data, admin_level, time_column)
        elif aggregation == "monthly":
            return self.daily_to_monthly(data, time_column)
        elif aggregation == "daily":
            logger.info("No temporal aggregation needed for daily data")
            return data
        else:
            raise ValueError(f"Unsupported temporal aggregation: {aggregation}")
