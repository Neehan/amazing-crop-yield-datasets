"""Soil data formatter for converting depth series to pivot format"""

import logging
from typing import Any

import pandas as pd

from src.processing.base.formatter import BaseFormatter

logger = logging.getLogger(__name__)


class SoilFormatter(BaseFormatter):
    """Soil-specific formatter for depth series data"""

    def _get_sorted_depth_columns(self, columns, variable_name: str) -> list:
        """Get depth columns sorted in proper depth order (not alphabetical)"""
        # Extract depth columns
        depth_cols = []
        for col in columns:
            col_str = str(col)
            if col_str.startswith(f"{variable_name}_"):
                depth_cols.append(col_str)

        # Define proper depth order
        depth_order = ["0_5cm", "5_15cm", "15_30cm", "30_60cm", "60_100cm", "100_200cm"]

        def depth_sort_key(col_name):
            """Extract depth part and return sort index"""
            parts = col_name.split("_")
            depth_part = "_".join(parts[-2:]) if len(parts) >= 2 else col_name
            return depth_order.index(depth_part)

        return sorted(depth_cols, key=depth_sort_key)

    def get_time_column_name(self) -> str:
        """Get the name of the time column for soil data"""
        return "depth"

    def get_time_grouping_columns(self, data: pd.DataFrame) -> dict:
        """Get the columns to group by for depth aggregation"""
        return {"year": 2020}

    def get_pivot_column_names(self, data: pd.DataFrame) -> dict:
        """Get mapping of depth ranges to output column names"""
        variable_name = data.iloc[0]["variable"]

        unique_depths = sorted(data["depth"].unique())

        column_mapping = {}
        for depth_key in unique_depths:
            column_mapping[depth_key] = f"{variable_name}_{depth_key}"

        return column_mapping

    def _validate_required_columns(self, data: pd.DataFrame) -> None:
        """Validate that all required columns are present"""
        required_cols = [
            "country_name",
            "admin_name",
            "admin_id",
            "latitude",
            "longitude",
            "depth",
            "variable",
        ]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")

    def _get_grouping_columns(self, data: pd.DataFrame, admin_level: int) -> list:
        """Get columns to group by for pivoting"""
        grouping_cols = ["country_name", "latitude", "longitude"]
        for level in range(1, admin_level + 1):
            if f"admin_level_{level}_name" in data.columns:
                grouping_cols.append(f"admin_level_{level}_name")
        return grouping_cols

    def _pivot_depth_data(
        self, data: pd.DataFrame, grouping_cols: list
    ) -> pd.DataFrame:
        """Pivot the data from depth series to depth columns"""
        if "value" not in data.columns:
            raise ValueError("Expected 'value' column not found in soil data")

        pivoted_table = data.pivot_table(
            index=grouping_cols, columns="depth", values="value", aggfunc="mean"
        )
        return pd.DataFrame(pivoted_table).reset_index()

    def _rename_depth_columns(
        self, pivoted: pd.DataFrame, variable_name: str, grouping_cols: list
    ) -> pd.DataFrame:
        """Rename depth columns to include variable name prefix"""
        depth_columns = [col for col in pivoted.columns if col not in grouping_cols]
        column_mapping = {
            depth_col: f"{variable_name}_{depth_col}" for depth_col in depth_columns
        }

        new_columns = {col: column_mapping.get(col, col) for col in pivoted.columns}
        pivoted.columns = [new_columns[col] for col in pivoted.columns]
        return pivoted

    def _build_final_column_order(
        self, pivoted: pd.DataFrame, variable_name: str, admin_level: int
    ) -> list:
        """Build the final column order for output"""
        final_cols = ["country_name"]
        for level in range(1, admin_level + 1):
            col_name = f"admin_level_{level}_name"
            if col_name in pivoted.columns:
                final_cols.append(col_name)
        final_cols.extend(["latitude", "longitude"])

        # Add depth columns in proper order
        depth_cols = self._get_sorted_depth_columns(pivoted.columns, variable_name)
        final_cols.extend(depth_cols)
        return final_cols

    def _apply_final_column_renaming(
        self, data: pd.DataFrame, admin_level: int
    ) -> pd.DataFrame:
        """Apply final column renaming for clean output"""
        rename_map = {"country_name": "country"}
        for level in range(1, admin_level + 1):
            old_name = f"admin_level_{level}_name"
            new_name = f"admin_level_{level}"
            if old_name in data.columns:
                rename_map[old_name] = new_name

        return data.rename(columns=rename_map)

    def pivot_to_final_format(
        self, data: pd.DataFrame, admin_level: int
    ) -> pd.DataFrame:
        """Convert depth series data to pivot format for CSV output"""
        logger.debug("Converting depth series data to pivot format")

        # Step 1: Validate input data
        self._validate_required_columns(data)
        df = data.copy()
        variable_name = df.iloc[0]["variable"]

        # Step 2: Get grouping columns and pivot data
        grouping_cols = self._get_grouping_columns(df, admin_level)
        pivoted = self._pivot_depth_data(df, grouping_cols)

        # Step 3: Rename depth columns with variable prefix
        pivoted = self._rename_depth_columns(pivoted, variable_name, grouping_cols)

        # Step 4: Build final column order and select columns
        final_cols = self._build_final_column_order(pivoted, variable_name, admin_level)
        available_cols = [col for col in final_cols if col in pivoted.columns]
        result_df = pd.DataFrame(pivoted[available_cols].copy())

        # Step 5: Apply final column renaming
        result_df = self._apply_final_column_renaming(result_df, admin_level)

        # Step 6: Sort by admin levels to match other data formats
        sort_columns = ["country"]
        for level in range(1, admin_level + 1):
            col_name = f"admin_level_{level}"
            if col_name in result_df.columns:
                sort_columns.append(col_name)

        result_df = result_df.sort_values(sort_columns).reset_index(drop=True)

        logger.debug(f"Soil pivot conversion complete: {result_df.shape}")
        return result_df
