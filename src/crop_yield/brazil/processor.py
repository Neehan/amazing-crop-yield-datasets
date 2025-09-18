"""
Brazil crop yield data processor.

Processes raw Brazil crop yield data and converts it into standardized CSV format.
"""

import pandas as pd
from pathlib import Path
import logging
import argparse
import geopandas as gpd

from src.crop_yield.brazil.models import CROP_NAME_MAPPING
from src.crop_yield.brazil.name_mapping import BrazilNameMapper
from src.crop_yield.base import filter_administrative_units_by_quality
from src.crop_yield.base.constants import (
    AREA_PLANTED_COLUMN,
    AREA_HARVESTED_COLUMN,
    PRODUCTION_COLUMN,
    YIELD_COLUMN,
)

logger = logging.getLogger(__name__)


def load_crop_data(crop_file: Path) -> pd.DataFrame:
    """Load and clean crop data from a single CSV file."""
    if not crop_file.exists():
        raise FileNotFoundError(f"Crop file not found: {crop_file}")

    # Read CSV
    df = pd.read_csv(crop_file)

    # Remove header row if it exists (first row with column names)
    if len(df) > 0 and df.iloc[0, 0] == "Nível Territorial (Código)":
        df = df.iloc[1:].copy()

    # Map columns based on SIDRA API response structure
    # D1C/D1N = municipality, D2C/D2N = year, D3C/D3N = variable, D4C/D4N = crop
    df = df[["D1C", "D1N", "D2C", "D2N", "D3C", "D3N", "D4C", "D4N", "V"]].copy()
    df.columns = [
        "municipality_code",
        "municipality",
        "year",
        "year_label",
        "variable_code",
        "variable",
        "crop_code",
        "crop",
        "value",
    ]

    # Keep all relevant variables
    relevant_variables = [
        "Rendimento médio da produção",
        "Área plantada",
        "Área colhida",
        "Quantidade produzida",
    ]
    df = df[df["variable"].isin(relevant_variables)]  # type: ignore

    # Convert value to numeric, handling missing values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Convert year to integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Map crop names to English
    df["crop_english"] = df["crop"].map(CROP_NAME_MAPPING)  # type: ignore

    # Remove rows with unmapped crops
    df = df.dropna(subset=["crop_english"])  # type: ignore

    return df


def load_all_crop_data(
    input_dir: Path, crop: str, start_year: int = 1974, end_year: int = 2024
) -> pd.DataFrame:
    """Load and merge crop data from all year files."""
    # Find all year files for this crop
    year_files = list(input_dir.glob(f"{crop}_*.csv"))

    if not year_files:
        raise FileNotFoundError(f"No year files found for {crop} in {input_dir}")

    logger.info(f"Found {len(year_files)} year files for {crop}")

    # Load and combine year files within the specified range
    all_data = []
    for year_file in sorted(year_files):
        # Extract year from filename (e.g., corn_2020.csv -> 2020)
        year = int(year_file.stem.split("_")[-1])
        if start_year <= year < end_year:
            year_data = load_crop_data(year_file)
            all_data.append(year_data)
            logger.debug(f"Loaded {len(year_data)} records from {year_file.name}")
        else:
            logger.warning(
                f"Skipped {year_file.name} (year {year} outside range {start_year}-{end_year-1})"
            )

    if not all_data:
        raise ValueError(
            f"No data found for {crop} in year range {start_year}-{end_year-1}"
        )

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined {len(df)} total records for {crop}")

    # Pivot data to have variables as columns
    df_pivot = df.pivot_table(
        index=["municipality_code", "municipality", "year", "crop_english"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Flatten column names
    df_pivot.columns.name = None

    # Rename columns to standard names
    column_mapping = {
        "Rendimento médio da produção": YIELD_COLUMN,
        "Área plantada": AREA_PLANTED_COLUMN,
        "Área colhida": AREA_HARVESTED_COLUMN,
        "Quantidade produzida": PRODUCTION_COLUMN,
    }
    df_pivot = df_pivot.rename(columns=column_mapping)

    # Ensure yield column exists (required for filtering)
    if YIELD_COLUMN not in df_pivot.columns:
        raise ValueError(f"No yield data found for {crop}")

    # Remove rows with missing or zero yield data
    initial_count = len(df_pivot)
    df_pivot = df_pivot.dropna(subset=[YIELD_COLUMN])
    df_pivot = df_pivot[df_pivot[YIELD_COLUMN] > 0]  # Remove zero yields
    final_count = len(df_pivot)

    if initial_count > final_count:
        logger.info(
            f"Dropped {initial_count - final_count} ({((initial_count - final_count) / initial_count * 100):.1f}%) rows with missing or zero yield"
        )

    return pd.DataFrame(df_pivot)


def process_crop_yield_data(
    input_dir: Path,
    output_dir: Path,
    crop: str,
    start_year: int = 1974,
    end_year: int = 2024,
    admin_level: int = 2,
) -> Path:
    """
    Process Brazil crop yield data and save to standardized format.

    Args:
        input_dir: Directory containing raw crop yield CSV files
        output_dir: Directory to save processed data
        crop: Crop name (e.g., "corn", "wheat")
        start_year: Starting year (inclusive, default: 1974)
        end_year: Ending year (exclusive, default: 2024)
        admin_level: Administrative level (1=state, 2=municipality)

    Returns:
        Path to the saved processed CSV file
    """
    # Load raw data from year files in specified range
    df = load_all_crop_data(input_dir, crop, start_year, end_year)

    logger.info(f"Loaded {len(df)} records for {crop}")

    # Load GADM data for name mapping
    gadm_file = Path("data/brazil/gadm/gadm41_BRA.gpkg")
    if not gadm_file.exists():
        raise FileNotFoundError(f"GADM file not found: {gadm_file}")

    boundaries = gpd.read_file(gadm_file, layer="ADM_ADM_2")
    gadm_admin1 = set(boundaries["NAME_1"].unique())
    gadm_admin2 = set(boundaries["NAME_2"].unique())

    # Initialize name mapper
    name_mapper = BrazilNameMapper(gadm_admin1, gadm_admin2)

    # Extract state from municipality name (last part after " - ")
    df["admin_level_1_raw"] = df["municipality"].str.extract(r" - ([A-Z]{2})$")

    # Use municipality as admin_level_2
    df["admin_level_2_raw"] = df["municipality"].str.replace(
        r" - [A-Z]{2}$", "", regex=True
    )

    # Map administrative names to GADM standard
    df["admin_level_1"] = df["admin_level_1_raw"].apply(
        lambda x: name_mapper.map_admin_name(x, 1) if pd.notna(x) else x
    )
    df["admin_level_2"] = df["admin_level_2_raw"].apply(
        lambda x: name_mapper.map_admin_name(x, 2) if pd.notna(x) else x
    )

    # Create standardized output
    output_data = {
        "country": "Brazil",
        "admin_level_1": df["admin_level_1"],
        "admin_level_2": df["admin_level_2"],
        "year": df["year"],
        f"{crop}_yield": df[YIELD_COLUMN],
    }

    # Add new columns if they exist in the data
    if AREA_PLANTED_COLUMN in df.columns:
        output_data[AREA_PLANTED_COLUMN] = df[AREA_PLANTED_COLUMN]
    if AREA_HARVESTED_COLUMN in df.columns:
        output_data[AREA_HARVESTED_COLUMN] = df[AREA_HARVESTED_COLUMN]
    if PRODUCTION_COLUMN in df.columns:
        output_data[PRODUCTION_COLUMN] = df[PRODUCTION_COLUMN]

    output_df = pd.DataFrame(output_data)

    # Remove rows with missing admin_level_1 (state)
    output_df = output_df.dropna(subset=["admin_level_1"])

    # Apply data quality filtering
    output_df = filter_administrative_units_by_quality(output_df, f"{crop}_yield")

    # Sort by admin_level_1, admin_level_2, year
    output_df = output_df.sort_values(
        ["admin_level_1", "admin_level_2", "year"]
    ).reset_index(drop=True)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed data with standardized filename
    output_file = output_dir / f"crop_{crop}_yield.csv"
    output_df.to_csv(output_file, index=False)

    logger.info(f"Processed data saved to {output_file}")
    logger.info(
        f"Final format: {len(output_df)} records with columns: {list(output_df.columns)}"
    )

    return output_file


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Process Brazil crop yield data")
    parser.add_argument(
        "--crop", required=True, help="Crop name (e.g., corn, wheat, soybean)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1974,
        help="Starting year (inclusive, default: 1974)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Ending year (exclusive, default: 2024)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/brazil/raw/crop_yield"),
        help="Input directory containing raw CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/brazil/final/crop"),
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

    try:
        output_file = process_crop_yield_data(
            args.input_dir,
            args.output_dir,
            args.crop,
            args.start_year,
            args.end_year,
            args.admin_level,
        )
        print(f"Successfully processed {args.crop} yield data")
        print(f"Processed data saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        raise


if __name__ == "__main__":
    main()
