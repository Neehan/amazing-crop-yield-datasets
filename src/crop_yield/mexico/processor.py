"""
Mexico crop yield data processor.

Processes raw Mexico crop yield HTML data from SIAP and converts it into standardized CSV format.
"""

import pandas as pd
from pathlib import Path
import logging
import argparse
import geopandas as gpd
import re
from io import StringIO
from typing import List

from src.crop_yield.mexico.models import CROP_NAME_MAPPING, COUNTRY_NAME
from src.crop_yield.mexico.name_mapping import MexicoNameMapper
from src.crop_yield.base import filter_administrative_units_by_quality
from src.crop_yield.base.constants import (
    AREA_PLANTED_COLUMN,
    AREA_HARVESTED_COLUMN,
    PRODUCTION_COLUMN,
    YIELD_COLUMN,
)

logger = logging.getLogger(__name__)


def parse_html_data(html_file: Path) -> pd.DataFrame:
    """Parse crop data from a single HTML file using pandas.read_html."""
    # Fail fast if file doesn't exist
    if not html_file.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    # Extract year from filename (e.g., corn_2024.html -> 2024)
    year = int(html_file.stem.split("_")[-1])

    # Read HTML file and extract content from CDATA
    with open(html_file, "r", encoding="iso-8859-1") as f:
        content = f.read()

    # Simple validation - just check basics to avoid false positives
    if len(content) < 1000:  # Very small files are likely errors
        raise ValueError(f"File too small ({len(content)} chars): {html_file}")

    # Extract the HTML content from CDATA section
    cdata_match = re.search(r"<!\[CDATA\[(.*?)\]\]>", content, re.DOTALL)
    if not cdata_match:
        raise ValueError(
            f"No CDATA section found in {html_file}. File may be corrupted."
        )

    html_content = cdata_match.group(1)

    # Extract crop name from content
    crop_match = re.search(r"Cultivo:\s*</strong>\s*([^(]+)", html_content)
    crop_name = crop_match.group(1).strip() if crop_match else "Unknown"

    # Use pandas to read the HTML table from the extracted content
    tables = pd.read_html(StringIO(html_content))

    if not tables:
        raise ValueError(f"No tables found in {html_file}")

    # Get the first (and likely only) table
    df = tables[0]

    # The table has a multi-level header, so we need to clean it up
    # Skip the first row which is just row numbers
    # Columns should be: [row_num, state, municipality, area_planted, area_harvested, area_damaged, production, yield, price, value]
    expected_cols = 10
    if len(df.columns) < expected_cols:
        raise ValueError(
            f"Table has {len(df.columns)} columns, expected {expected_cols}"
        )

    # Set proper column names
    df.columns = [
        "row_num",
        "state",
        "municipality",
        "area_planted",
        "area_harvested",
        "area_damaged",
        "production",
        "yield",
        "price",
        "value",
    ]

    # Remove header rows - keep only numeric row numbers
    df = df[pd.to_numeric(df["row_num"], errors="coerce").notnull()].copy()  # type: ignore

    # Convert numeric columns, handling commas as thousands separators
    numeric_cols = [
        "area_planted",
        "area_harvested",
        "area_damaged",
        "production",
        "yield",
        "price",
        "value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Clean and convert numeric columns
            series = df[col].astype(str)
            series = series.str.replace(",", "", regex=False)  # type: ignore
            series = series.str.replace(" ", "", regex=False)  # type: ignore
            numeric_series = pd.to_numeric(series, errors="coerce")
            df[col] = numeric_series.fillna(0.0)  # type: ignore

    # Add metadata
    df["crop"] = crop_name
    df["year"] = year

    # Clean string columns
    df["state"] = df["state"].astype(str).str.strip()  # type: ignore
    df["municipality"] = df["municipality"].astype(str).str.strip()  # type: ignore

    if len(df) == 0:
        raise ValueError(f"No data extracted from {html_file}")

    # Map crop names to English
    df["crop_english"] = df["crop"].map(CROP_NAME_MAPPING)  # type: ignore

    # Remove rows with unmapped crops
    df_clean = df.dropna(subset=["crop_english"])  # type: ignore

    logger.debug(f"Extracted {len(df_clean)} records from {html_file.name}")
    return pd.DataFrame(df_clean)


def load_all_crop_data(
    input_dir: Path, crops: List[str], start_year: int = 2003, end_year: int = 2024
) -> pd.DataFrame:
    """Load and merge crop data from all HTML year files for multiple crops."""
    from src.crop_yield.mexico.models import CROP_CODES

    # If no crops specified, use all available crops
    if not crops:
        crops = list(CROP_CODES.keys())
        logger.info(
            f"No crops specified, processing all available crops: {', '.join(crops)}"
        )

    # Validate all crop names - fail fast if any invalid
    for crop in crops:
        if crop not in CROP_CODES:
            raise ValueError(
                f"Unknown crop: {crop}. Available crops: {list(CROP_CODES.keys())}"
            )

    # Generate list of expected files based on year range
    years = list(range(start_year, end_year))
    logger.info(
        f"Loading {len(crops)} crops data for {len(years)} years ({start_year}-{end_year-1})"
    )

    # Track missing and invalid files for comprehensive reporting
    missing_files = []
    invalid_files = []
    successful_files = []
    all_data = []

    # Load and combine all crop-year files
    for crop in crops:
        crop_missing = []
        for year in years:
            year_file = input_dir / f"{crop}_{year}.html"
            if year_file.exists():
                try:
                    year_data = parse_html_data(year_file)
                    if not year_data.empty:
                        all_data.append(year_data)
                        successful_files.append(year_file.name)
                        logger.debug(
                            f"Loaded {len(year_data)} records from {year_file.name}"
                        )
                    else:
                        logger.warning(
                            f"No valid data extracted from {year_file.name} - file may contain server error or no crop data for this year"
                        )
                        invalid_files.append(year_file.name)
                except Exception as e:
                    logger.error(f"Failed to parse {year_file.name}: {e}")
                    invalid_files.append(year_file.name)
            else:
                missing_files.append(year_file.name)
                crop_missing.append(year)

        # Report missing years for each crop
        if crop_missing:
            missing_years = sorted(crop_missing)
            logger.warning(
                f"Missing {len(missing_years)} years for {crop}: {missing_years}"
            )

    # Comprehensive reporting
    total_expected = len(crops) * len(years)
    logger.info(f"File processing summary:")
    logger.info(f"  Expected files: {total_expected}")
    logger.info(f"  Successful: {len(successful_files)}")
    logger.info(f"  Missing: {len(missing_files)}")
    logger.info(f"  Invalid/corrupted: {len(invalid_files)}")

    if missing_files:
        logger.warning(f"Missing files may indicate download failures or data gaps")
    if invalid_files:
        logger.warning(f"Invalid files may indicate server errors during download")

    # Fail fast if no valid data found
    if not all_data:
        raise ValueError(
            f"No valid data found for crops {crops} in year range {start_year}-{end_year-1}. "
            f"Missing: {len(missing_files)} files, Invalid: {len(invalid_files)} files."
        )

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined {len(df)} total records for {len(crops)} crops")

    # Remove rows with missing or zero yield data
    initial_count = len(df)
    df = df.dropna(subset=["yield"])
    df = df[df["yield"] > 0]  # Remove zero yields
    df["yield"] *= 1000.0  # Convert to kg/ha
    final_count = len(df)

    if initial_count > final_count:
        logger.info(
            f"Dropped {initial_count - final_count} ({((initial_count - final_count) / initial_count * 100):.1f}%) rows with missing or zero yield"
        )

    return pd.DataFrame(df)


def process_crop_yield_data(
    input_dir: Path,
    output_dir: Path,
    crops: List[str],
    start_year: int = 2003,
    end_year: int = 2024,
    admin_level: int = 2,
) -> List[Path]:
    """
    Process Mexico crop yield data and save to standardized format.

    Args:
        input_dir: Directory containing raw HTML files
        output_dir: Directory to save processed data
        crops: List of crop names (e.g., ["corn", "soybean"]) or empty list for all crops
        start_year: Starting year (inclusive, default: 2003)
        end_year: Ending year (exclusive, default: 2024)
        admin_level: Administrative level (1=state, 2=municipality)

    Returns:
        List of paths to the saved processed CSV files
    """
    # Load raw data from HTML files in specified range
    df = load_all_crop_data(input_dir, crops, start_year, end_year)

    logger.info(f"Loaded {len(df)} total records for all crops")

    # Load GADM data for name mapping - fail fast if required file missing
    gadm_file = Path("data/mexico/gadm/gadm41_MEX.gpkg")
    if not gadm_file.exists():
        raise FileNotFoundError(
            f"GADM file not found: {gadm_file}. Download required for processing."
        )

    # Load GADM boundaries for name mapping
    boundaries = gpd.read_file(gadm_file, layer="ADM_ADM_2")
    gadm_admin1 = set(boundaries["NAME_1"].unique())
    gadm_admin2 = set(boundaries["NAME_2"].unique())

    # Initialize name mapper
    name_mapper = MexicoNameMapper(gadm_admin1, gadm_admin2)

    # Check for unmapped names before processing
    html_states = set(df["state"].unique())
    html_munis = set(df["municipality"].unique())
    unmapped_states = html_states - gadm_admin1
    unmapped_munis = html_munis - gadm_admin2

    logger.info(
        f"Name mapping: {len(html_states)} states, {len(html_munis)} municipalities"
    )
    logger.info(
        f"Unmapped: {len(unmapped_states)} states, {len(unmapped_munis)} municipalities"
    )

    # Map administrative names to GADM standard
    df["admin_level_1"] = df["state"].apply(
        lambda x: name_mapper.map_admin_name(x, 1) if pd.notna(x) else x
    )
    df["admin_level_2"] = df["municipality"].apply(
        lambda x: name_mapper.map_admin_name(x, 2) if pd.notna(x) else x
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each crop separately to create individual CSV files
    output_files = []
    unique_crops = df["crop_english"].unique()

    for crop in unique_crops:
        if pd.isna(crop):
            continue

        # Filter data for this crop
        crop_df = df[df["crop_english"] == crop].copy()
        logger.info(f"Processing {len(crop_df)} records for {crop}")

        # Create standardized output matching Brazil format exactly
        output_data = {
            "country": COUNTRY_NAME,
            "admin_level_1": crop_df["admin_level_1"],
            "admin_level_2": crop_df["admin_level_2"],
            "year": crop_df["year"],
            YIELD_COLUMN: crop_df["yield"],  # Generic yield column
            AREA_PLANTED_COLUMN: crop_df["area_planted"],
            AREA_HARVESTED_COLUMN: crop_df["area_harvested"],
            PRODUCTION_COLUMN: crop_df["production"],
        }

        output_df = pd.DataFrame(output_data)

        # Remove rows with missing admin_level_1 (state)
        output_df = output_df.dropna(subset=["admin_level_1"])

        # Apply data quality filtering
        output_df = filter_administrative_units_by_quality(output_df, YIELD_COLUMN)

        # Sort by admin_level_1, admin_level_2, year
        output_df = output_df.sort_values(
            ["admin_level_1", "admin_level_2", "year"]
        ).reset_index(drop=True)

        # Save processed data with standardized filename
        output_file = output_dir / f"crop_{crop}_yield.csv"
        output_df.to_csv(output_file, index=False)
        output_files.append(output_file)

        logger.info(f"Processed data for {crop} saved to {output_file}")
        logger.info(
            f"Final format: {len(output_df)} records with columns: {list(output_df.columns)}"
        )

    return output_files


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Process Mexico crop yield data")
    parser.add_argument(
        "--crops",
        nargs="*",
        help="Crop names (e.g., corn soybean). If not specified, processes all available crops",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1980,
        help="Starting year (inclusive, default: 1980)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Ending year (exclusive, default: 2025)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/mexico/raw/crop_yield"),
        help="Input directory containing raw HTML files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mexico/final/crop"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--admin-level",
        type=int,
        default=2,
        choices=[1, 2],
        help="Administrative level (1=state, 2=municipality)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Default to empty list if no crops specified (will process all)
    crops = args.crops or []

    output_files = process_crop_yield_data(
        args.input_dir,
        args.output_dir,
        crops,
        args.start_year,
        args.end_year,
        args.admin_level,
    )
    print(f"Successfully processed {len(output_files)} crop yield datasets")
    print(f"Processed data saved to: {args.output_dir}")
    for output_file in output_files:
        print(f"  - {output_file.name}")


if __name__ == "__main__":
    main()
