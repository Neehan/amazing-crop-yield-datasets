"""
Argentina crop yield data processor.

Simple processor that converts raw Argentina crop yield data into standardized CSV format.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse

from src.crop_yield.argentina.models import (
    CROP_NAME_MAPPING,
    DATA_QUALITY_THRESHOLD,
    EVALUATION_YEARS,
    RAW_DATA_COLUMNS,
    COUNTRY_NAME,
)
from src.crop_yield.argentina.name_mapping import ArgentinaNameMapper

logger = logging.getLogger(__name__)


def extract_harvest_year(season: str) -> int:
    """Extract harvest year from season string like '2023/24' -> 2024."""
    if "/" not in season:
        raise ValueError(f"Invalid season format: {season}")

    parts = season.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid season format: {season}")

    start_year = int(parts[0])
    return start_year + 1


def load_crop_data(crop_file: Path) -> pd.DataFrame:
    """Load and clean crop data from CSV file."""
    if not crop_file.exists():
        raise FileNotFoundError(f"Crop file not found: {crop_file}")

    # Read CSV with semicolon separator
    df = pd.read_csv(crop_file, sep=";", encoding="latin1")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', "")

    # Rename columns to standard names
    column_mapping = {
        raw: std for std, raw in RAW_DATA_COLUMNS.items() if raw in df.columns
    }
    if not column_mapping:
        raise ValueError(f"No recognized columns found in {crop_file}")

    df = df.rename(columns=column_mapping)

    # Extract harvest year
    df["year"] = df["season"].apply(extract_harvest_year)

    # Clean data types
    for col in ["crop", "province", "department"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', "")

    if "yield" in df.columns:
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    else:
        raise ValueError("Yield column not found in the data")

    # Drop rows where yield is missing or zero
    initial_count = len(df)
    df = df.dropna(subset=["yield"])  # Remove missing
    df = df[df["yield"] > 0]  # Remove zero yields
    final_count = len(df)

    if initial_count > final_count:
        logger.info(
            f"Dropped {initial_count - final_count} ({((initial_count - final_count) / initial_count * 100):.1f}%) rows with missing or zero yield"
        )

    return pd.DataFrame(df)


def filter_departments_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Filter departments with insufficient data quality."""
    max_year = df["year"].max()
    recent_years = range(max_year - EVALUATION_YEARS + 1, max_year + 1)
    recent_data = df[df["year"].isin(recent_years)]

    if len(recent_data) == 0:
        raise ValueError(f"No data in evaluation period")

    # Calculate completeness by department
    dept_stats = []
    grouped = recent_data.groupby(["province", "department"])
    for group_key, group_data in grouped:
        prov, dept = group_key[0], group_key[1]  # type: ignore
        completeness = group_data["yield"].notna().sum() / len(recent_years)  # type: ignore
        if completeness >= DATA_QUALITY_THRESHOLD:
            dept_stats.append({"province": prov, "department": dept})

    if not dept_stats:
        raise ValueError(
            f"No departments meet {DATA_QUALITY_THRESHOLD:.0%} quality threshold"
        )

    # Keep only good departments
    good_depts = pd.DataFrame(dept_stats)
    filtered_df = df.merge(good_depts, on=["province", "department"], how="inner")

    total_depts = len(recent_data.groupby(["province", "department"]))
    kept_depts = len(good_depts)
    drop_pct = (
        ((total_depts - kept_depts) / total_depts * 100) if total_depts > 0 else 0
    )

    logger.info(
        f"Departments: kept {kept_depts}/{total_depts} ({drop_pct:.1f}% dropped)"
    )

    return filtered_df


def process_crop_file(crop_file: Path, output_dir: Path) -> Path:
    """Process a single crop file."""
    # Load GADM names for mapping
    import geopandas as gpd

    boundaries = gpd.read_file("data/argentina/gadm/gadm41_ARG.gpkg", layer="ADM_ADM_2")
    gadm_admin1 = set(boundaries["NAME_1"].unique())
    gadm_admin2 = set(boundaries["NAME_2"].unique())

    # Get crop name from file path (e.g. wheat.csv -> wheat)
    crop_english = crop_file.stem  # Gets filename without extension
    logger.info(f"Processing {crop_english}")

    # Initialize name mapper
    name_mapper = ArgentinaNameMapper(gadm_admin1, gadm_admin2)

    # Load data
    df = load_crop_data(crop_file)
    # Calculate missing data stats on RAW data before any cleaning
    raw_df = pd.read_csv(crop_file, sep=";", encoding="latin1")
    raw_df.columns = raw_df.columns.str.strip().str.replace('"', "")
    column_mapping = {
        raw: std for std, raw in RAW_DATA_COLUMNS.items() if raw in raw_df.columns
    }
    raw_df = raw_df.rename(columns=column_mapping)
    if "yield" in raw_df.columns:
        raw_df["yield"] = pd.to_numeric(raw_df["yield"], errors="coerce")

    total_records_original = len(raw_df)
    missing_yield_original = raw_df["yield"].isna().sum()
    zero_yield_original = (raw_df["yield"] == 0).sum()
    bad_yield_original = missing_yield_original + zero_yield_original
    missing_pct_original = (
        (bad_yield_original / total_records_original * 100)
        if total_records_original > 0
        else 0
    )

    # Filter by quality
    df_filtered = filter_departments_by_quality(df)

    # Map administrative names to GADM standard
    df_filtered["admin_level_1"] = df_filtered["province"].apply(
        lambda x: name_mapper.map_admin_name(x, 1)
    )
    df_filtered["admin_level_2"] = df_filtered["department"].apply(
        lambda x: name_mapper.map_admin_name(x, 2)
    )

    # Create output
    output_df = pd.DataFrame(
        {
            "country": COUNTRY_NAME,
            "admin_level_1": df_filtered["admin_level_1"],
            "admin_level_2": df_filtered["admin_level_2"],
            "year": df_filtered["year"],
            f"{crop_english}_yield": df_filtered["yield"],
        }
    )

    # Sort and save
    output_df = output_df.sort_values(["admin_level_1", "admin_level_2", "year"])

    output_file = output_dir / f"crop_{crop_english}_yield.csv"
    output_df.to_csv(output_file, index=False)

    final_departments = (
        output_df[["admin_level_1", "admin_level_2"]].drop_duplicates().shape[0]
    )

    min_year = output_df["year"].min()
    max_year = output_df["year"].max()
    logger.info(
        f"Processed {crop_english}: {len(output_df):,} records ({min_year}-{max_year}), "
        f"{final_departments} departments, {missing_pct_original:.2f}% missing/zero yield in raw data"
    )

    return output_file


def process_all_crops(
    data_dir: Path, output_dir: Path, crop_filter: Optional[List[str]] = None
) -> List[Path]:
    """Process all crop files or specific crops if filtered."""
    # Build crop file list from models
    crops_to_process = crop_filter if crop_filter else list(CROP_NAME_MAPPING.values())

    crop_files = []
    for crop_name in crops_to_process:
        crop_file = data_dir / f"{crop_name}.csv"
        if not crop_file.exists():
            raise FileNotFoundError(f"Crop file not found: {crop_file}")
        crop_files.append(crop_file)

    logger.info(f"Processing {len(crop_files)} crop files")

    output_files = []
    failed_crops = []

    for crop_file in crop_files:
        try:
            output_file = process_crop_file(crop_file, output_dir)
            output_files.append(output_file)
        except Exception as e:
            logger.error(f"Failed to process {crop_file.name}: {e}")
            failed_crops.append(crop_file.name)

    if failed_crops:
        logger.warning(f"Failed crops: {failed_crops}")

    if not output_files:
        raise RuntimeError("No crops were successfully processed")

    return output_files


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Process Argentina crop yield data")
    parser.add_argument(
        "--crop",
        nargs="+",
        help="Specific crops to process (e.g., wheat corn). If not provided, all crops are processed.",
        choices=list(CROP_NAME_MAPPING.values()),
        metavar="CROP",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with detailed logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    data_dir = Path("data/argentina/raw/crop_yield")
    output_dir = Path("data/argentina/final")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = process_all_crops(data_dir, output_dir, args.crop)
    logger.info(f"Successfully processed {len(output_files)} crops")

    for file_path in output_files:
        logger.debug(f"Output: {file_path}")


if __name__ == "__main__":
    main()
