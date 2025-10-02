"""
Data quality filtering for crop yield data.

Common filtering logic used across different countries to ensure data quality
by removing administrative units with insufficient data coverage.
"""

import pandas as pd
import logging

from src.crop_yield.base.constants import DATA_QUALITY_THRESHOLD, EVALUATION_YEARS

logger = logging.getLogger(__name__)


def filter_administrative_units_by_quality(
    df: pd.DataFrame,
    yield_column: str,
    admin_level_1_col: str = "admin_level_1",
    admin_level_2_col: str = "admin_level_2",
) -> pd.DataFrame:
    """
    Filter administrative units with insufficient data quality.

    Args:
        df: DataFrame with crop yield data
        yield_column: Name of the yield column
        admin_level_1_col: Name of the admin level 1 column (e.g., "province", "state")
        admin_level_2_col: Name of the admin level 2 column (e.g., "department", "municipality")

    Returns:
        Filtered DataFrame with only high-quality administrative units
    """
    max_year = df["year"].max()
    recent_years = range(max_year - EVALUATION_YEARS + 1, max_year + 1)  # type: ignore
    recent_data = df[df["year"].isin(recent_years)]

    if len(recent_data) == 0:
        raise ValueError(f"No data in evaluation period")

    # Calculate completeness by administrative unit
    admin_stats = []
    grouped = recent_data.groupby([admin_level_1_col, admin_level_2_col])
    for group_key, group_data in grouped:
        admin1, admin2 = group_key[0], group_key[1]  # type: ignore
        completeness = group_data[yield_column].notna().sum() / len(recent_years)  # type: ignore
        if completeness >= DATA_QUALITY_THRESHOLD:
            admin_stats.append({admin_level_1_col: admin1, admin_level_2_col: admin2})

    if not admin_stats:
        raise ValueError(
            f"No administrative units meet {DATA_QUALITY_THRESHOLD:.0%} quality threshold"
        )

    # Keep only good administrative units
    good_units = pd.DataFrame(admin_stats)
    filtered_df = df.merge(
        good_units, on=[admin_level_1_col, admin_level_2_col], how="inner"
    )

    total_units = len(recent_data.groupby([admin_level_1_col, admin_level_2_col]))
    kept_units = len(good_units)
    drop_pct = (
        ((total_units - kept_units) / total_units * 100) if total_units > 0 else 0
    )

    logger.info(
        f"Administrative units: kept {kept_units}/{total_units} ({drop_pct:.1f}% dropped)"
    )

    return filtered_df
