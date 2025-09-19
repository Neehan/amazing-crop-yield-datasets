"""Base formatter for converting time series data to output formats"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm

# Dynamic admin level handling - no hardcoded constants needed

logger = logging.getLogger(__name__)


class BaseFormatter(ABC):
    """Base class for formatting time series data to output formats"""

    def __init__(self):
        """Initialize base formatter"""
        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def get_time_column_name(self) -> str:
        """Get the name of the time column for this data type"""
        pass

    @abstractmethod
    def get_time_grouping_columns(self, data: pd.DataFrame) -> dict:
        """Get the columns to group by for time aggregation"""
        pass

    @abstractmethod
    def get_pivot_column_names(self, data: pd.DataFrame) -> dict:
        """Get mapping of time values to output column names"""
        pass

    def pivot_to_final_format(
        self, data: pd.DataFrame, admin_level: int, output_file_path: Path
    ) -> pd.DataFrame:
        """Convert time series data to pivot format for CSV output

        Args:
            data: DataFrame with time series data
            admin_level: GADM admin level being processed
            output_file_path: Path where final output will be saved

        Returns:
            DataFrame with pivot format
        """

        # Check if output already exists - skip expensive formatting if so
        if output_file_path.exists():
            logger.info(f"Output file already exists, loading from cache: {output_file_path}")
            return pd.read_csv(output_file_path)

        logger.debug(f"Output file doesn't exist, proceeding with formatting: {output_file_path}")
        logger.debug("Converting time series data to pivot format")

        time_column = self.get_time_column_name()
        assert (
            time_column in data.columns
        ), f"Time column '{time_column}' not found in data"

        # Ensure time column is datetime
        df = data.copy()
        df[time_column] = pd.to_datetime(df[time_column])

        # Get time grouping columns (e.g., year, week)
        grouping_columns = self.get_time_grouping_columns(df)

        # Add grouping columns to dataframe
        for col, values in grouping_columns.items():
            df[col] = values

        # Get the value column and variable name
        admin_cols = [
            "country_name",
            "admin_name",
            "admin_id",
            "latitude",
            "longitude",
            time_column,
            "variable",
        ]

        # Add dynamic admin level columns based on the target admin level
        for level in range(1, admin_level + 1):
            admin_cols.append(f"admin_level_{level}_name")

        admin_cols.extend(list(grouping_columns.keys()))

        value_cols = [col for col in df.columns if col not in admin_cols]
        assert (
            len(value_cols) == 1
        ), f"Expected 1 value column, got {len(value_cols)}: {value_cols}"
        value_col = value_cols[0]

        # Get variable name for column naming
        assert "variable" in df.columns, "Variable column required for proper naming"
        variable_name = df.iloc[0]["variable"]

        # Build grouping columns based on admin level
        pivot_grouping_cols = ["country_name", "latitude", "longitude"]
        for level in range(1, admin_level + 1):
            pivot_grouping_cols.append(f"admin_level_{level}_name")

        # Add time grouping columns (e.g., year)
        pivot_grouping_cols.extend(
            [col for col in grouping_columns.keys() if col != "time_unit"]
        )

        # Get pivot column (e.g., week_num)
        pivot_col = [col for col in grouping_columns.keys() if col == "time_unit"]
        assert len(pivot_col) == 1, f"Expected 1 pivot column, got {len(pivot_col)}"
        pivot_col = pivot_col[0]

        # Group and aggregate
        tqdm.pandas(desc="Grouping time series data")
        aggregated = (
            df.groupby(pivot_grouping_cols + [pivot_col])[value_col]
            .progress_apply(lambda x: x.mean())
            .reset_index()
        )

        # Build index columns for pivot
        index_cols = ["country_name"]
        for level in range(1, admin_level + 1):
            index_cols.append(f"admin_level_{level}_name")
        index_cols.extend(["latitude", "longitude"])
        index_cols.extend(
            [col for col in grouping_columns.keys() if col != "time_unit"]
        )

        # Pivot: rows = admin hierarchy + time grouping, columns = time_unit, values = weather value
        pivoted_table = aggregated.pivot_table(
            index=index_cols, columns=pivot_col, values=value_col, aggfunc="mean"
        )
        pivoted = pd.DataFrame(pivoted_table).reset_index()

        # Get column name mapping
        column_mapping = self.get_pivot_column_names(df)

        # Rename columns
        new_columns = {}
        for col in pivoted.columns:
            new_columns[col] = column_mapping.get(col, col)
        pivoted.columns = [new_columns[col] for col in pivoted.columns]

        # Fill missing time periods with NaN
        expected_columns = list(column_mapping.values())
        for col in expected_columns:
            if col not in pivoted.columns:
                pivoted[col] = np.nan

        # Preserve column order - admin levels right after country
        final_cols = ["country_name"]
        for level in range(1, admin_level + 1):
            final_cols.append(f"admin_level_{level}_name")
        final_cols.extend(
            [col for col in grouping_columns.keys() if col != "time_unit"]
        )
        final_cols.extend(["latitude", "longitude"])
        # Use original order from column_mapping instead of sorting
        final_cols.extend([col for col in expected_columns if col in pivoted.columns])

        # Exclude week 53 columns if they exist
        week_53_cols = [col for col in final_cols if col.endswith("_week_53")]
        if week_53_cols:
            final_cols = [col for col in final_cols if not col.endswith("_week_53")]

        # Reorder columns
        pivoted = pivoted[final_cols]

        # Rename columns to match expected output format
        rename_map = {"country_name": "country"}
        for level in range(1, admin_level + 1):
            rename_map[f"admin_level_{level}_name"] = f"admin_level_{level}"

        # Apply final column renaming
        new_columns = {}
        for col in pivoted.columns:
            new_columns[col] = rename_map.get(col, col)
        pivoted.columns = [new_columns[col] for col in pivoted.columns]

        logger.debug(f"Pivot conversion complete: {pivoted.shape}")
        return pd.DataFrame(pivoted)
