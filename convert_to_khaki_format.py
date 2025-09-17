#!/usr/bin/env python3
"""
Convert processed data to Khaki format for crop yield prediction.

This script takes country and crop parameters and converts the processed data
to the format expected by the Khaki yield dataloader.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_concatenate_features(data_dir: str, country: str) -> pd.DataFrame:
    """Load and concatenate all feature chunks for a country."""
    features_dir = Path(data_dir) / country / "final" / "features"

    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    # Find all feature chunk files
    feature_files = sorted(features_dir.glob("features_chunk_*.csv"))

    if not feature_files:
        raise FileNotFoundError(f"No feature chunk files found in {features_dir}")

    logger.info(f"Found {len(feature_files)} feature chunk files")

    # Load and concatenate all chunks
    feature_dfs = []
    for file_path in feature_files:
        logger.info(f"Loading {file_path.name}...")
        df = pd.read_csv(file_path)
        feature_dfs.append(df)

    # Concatenate all feature data
    features_df = pd.concat(feature_dfs, ignore_index=True)
    logger.info(f"Concatenated features shape: {features_df.shape}")

    return features_df


def load_yield_data(data_dir: str, country: str, crops: List[str]) -> pd.DataFrame:
    """Load yield data for multiple crops and merge them."""
    yield_dfs = []

    for crop in crops:
        yield_file = (
            Path(data_dir) / country / "final" / "crop" / f"crop_{crop}_yield.csv"
        )
        if yield_file.exists():
            yield_df = pd.read_csv(yield_file)
            yield_df.drop(
                columns=["area_planted", "area_harvested", "production"], inplace=True
            )
            yield_dfs.append(yield_df)
            logger.info(f"Loaded yield data for {crop}: {yield_df.shape}")

    if not yield_dfs:
        raise FileNotFoundError(f"No yield data found for crops: {crops}")

    # Merge all yield dataframes
    merged_yield_df = yield_dfs[0]
    for yield_df in yield_dfs[1:]:
        merged_yield_df = merged_yield_df.merge(
            yield_df,
            on=["country", "admin_level_1", "admin_level_2", "year"],
            how="outer",
        )

    return merged_yield_df


def create_location_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Create unique integer location IDs from admin1/admin2 combinations."""
    # Create a mapping of unique admin combinations to integer IDs
    admin_combinations = df[["admin_level_1", "admin_level_2"]].drop_duplicates()
    admin_combinations = admin_combinations.reset_index(drop=True)
    admin_combinations["loc_ID"] = admin_combinations.index + 1

    # Merge back to get loc_ID for all rows
    df = df.merge(admin_combinations, on=["admin_level_1", "admin_level_2"], how="left")

    logger.info(f"Created {len(admin_combinations)} unique location IDs")
    return df


def map_weather_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map weather variables to Khaki format.

    Khaki expects W1-W6 format where:
    - W1: precipitation (week 1-52) mm / day
    - W2: solar radiation (week 1-52) in MJ/m²/day
    - W3: snow depth (week 1-52) cm of water equivalent
    - W4: max temperature (week 1-52) in Celsius
    - W5: min temperature (week 1-52) in Celsius
    - W6: vapor pressure (week 1-52)
    """
    logger.info("Mapping weather variables to Khaki format...")

    # Weather variable mapping
    weather_mapping = {
        "precipitation": "W1",
        "solar_radiation": "W2",
        "snow_lwe": "W3",
        "t2m_max": "W4",
        "t2m_min": "W5",
        "vapor_pressure": "W6",
    }

    # Create new weather columns in Khaki format
    weather_data = {}
    for week in range(1, 53):
        for weather_var, khaki_prefix in weather_mapping.items():
            old_col = f"{weather_var}_week_{week}"
            new_col = f"W_{khaki_prefix[1:]}_{week}"

            if old_col in df.columns:
                data = df[old_col].copy()

                # Apply unit conversions
                if weather_var in ["t2m_max", "t2m_min"]:
                    # Convert temperature from Kelvin to Celsius
                    data = data - 273.15
                    logger.debug(f"Converted {old_col} from Kelvin to Celsius")
                elif weather_var == "solar_radiation":
                    # Convert solar radiation from J/m²/day to MJ/m²/day
                    data = data / 1_000_000
                    logger.debug(f"Converted {old_col} from J/m²/day to MJ/m²/day")
                elif weather_var == "vapor_pressure":
                    # Convert vapor pressure from hPa to kPa
                    data = data / 10.0
                    logger.debug(f"Converted {old_col} from hPa to kPa")

                weather_data[new_col] = data
            else:
                logger.warning(f"Column {old_col} not found in data")
                weather_data[new_col] = 0.0

    # Add all weather columns at once to avoid fragmentation
    weather_df = pd.DataFrame(weather_data, index=df.index)
    df = pd.concat([df, weather_df], axis=1)

    return df


def map_soil_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map soil variables to Khaki format.

    Khaki expects soil variables in format: {measure}_mean_{depth}
    We have 10 soil measurements available, 1 missing (ocs):
    Available: bulk_density->bdod, cec, clay, coarse_fragments->cfvo, nitrogen,
              organic_carbon->soc, organic_carbon_density->ocd, ph_h2o->phh2o, sand, silt
    Missing: ocs (set to zero)
    """
    logger.info("Mapping soil variables to Khaki format...")

    # Mapping from Khaki names to our data column names
    soil_mapping = {
        "bdod": "bulk_density",
        "cec": "cec",
        "clay": "clay",
        "cfvo": "coarse_fragments",
        "nitrogen": "nitrogen",
        "soc": "organic_carbon",
        "ocd": "organic_carbon_density",
        "phh2o": "ph_h2o",
        "sand": "sand",
        "silt": "silt",
        "ocs": None,  # Missing - set to zero
    }
    soil_depths = ["0_5cm", "5_15cm", "15_30cm", "30_60cm", "60_100cm", "100_200cm"]

    # Create soil columns in Khaki format
    soil_data = {}
    for khaki_measure, data_measure in soil_mapping.items():
        for depth in soil_depths:
            new_col = f"{khaki_measure}_mean_{depth.replace('_', '-')}"

            if data_measure is not None:
                old_col = f"{data_measure}_{depth}"
                if old_col in df.columns:
                    soil_data[new_col] = df[old_col]
                else:
                    logger.warning(f"Soil column {old_col} not found, setting to zero")
                    soil_data[new_col] = 0.0
            else:
                # Missing measurement - set to zero
                logger.warning(f"Missing soil measurement: {new_col}")
                soil_data[new_col] = 0.0

    # Add all soil columns at once to avoid fragmentation
    soil_df = pd.DataFrame(soil_data, index=df.index)
    df = pd.concat([df, soil_df], axis=1)

    return df


def add_practice_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add practice columns (P1-P14) all set to zero."""
    logger.info("Adding practice columns (P1-P14) set to zero...")

    # Add all practice columns at once to avoid fragmentation
    practice_data = {f"P_{i}": 0.0 for i in range(1, 15)}
    practice_df = pd.DataFrame(practice_data, index=df.index)
    df = pd.concat([df, practice_df], axis=1)

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match Khaki format."""
    logger.info("Renaming columns to Khaki format...")

    # Rename admin columns
    df = df.rename(
        columns={
            "admin_level_1": "State",
            "admin_level_2": "County",
            "latitude": "lat",
            "longitude": "lng",
        }
    )

    return df


def convert_to_khaki_format(
    data_dir: str, country: str, crops: List[str], output_file: Optional[str] = None
) -> str:
    """
    Convert processed data to Khaki format.

    Args:
        data_dir: Path to data directory
        country: Country name (e.g., 'argentina')
        crop: Crop type (e.g., 'soybean')
        output_file: Output file path (optional)

    Returns:
        Path to the output file
    """
    logger.info(f"Converting {country} {crops} data to Khaki format...")

    # Load and concatenate features
    features_df = load_and_concatenate_features(data_dir, country)

    # Load yield data
    yield_df = load_yield_data(data_dir, country, crops)

    # Merge features with yield data (left join to preserve rows without yield as NaN)
    logger.info("Merging features with yield data...")
    merged_df = features_df.merge(
        yield_df, on=["country", "admin_level_1", "admin_level_2", "year"], how="left"
    )
    logger.info(f"After merging: {merged_df.shape[0]} rows")

    # Create unique location IDs
    merged_df = create_location_ids(merged_df)

    # Map weather variables
    merged_df = map_weather_variables(merged_df)

    # Map soil variables
    merged_df = map_soil_variables(merged_df)

    # Add practice columns
    merged_df = add_practice_columns(merged_df)

    # Rename columns
    merged_df = rename_columns(merged_df)

    # Select and reorder columns to match Khaki format
    # Start with basic columns in the correct order
    khaki_columns = ["State", "County", "year", "lat", "lng", "loc_ID"]

    # Add yield columns for all crops
    yield_columns = [col for col in merged_df.columns if col.endswith("_yield")]
    if not yield_columns:
        raise ValueError("No yield columns found")

    # Convert yield columns to integers (NaN values will remain as NaN)
    for col in yield_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="raise").astype("Int64")  # type: ignore

    khaki_columns.extend(yield_columns)

    # Add weather columns (W_1_1 to W_6_52)
    for w in range(1, 7):
        for week in range(1, 53):
            khaki_columns.append(f"W_{w}_{week}")

    # Add soil columns
    soil_measurements = [
        "bdod",
        "cec",
        "cfvo",
        "clay",
        "nitrogen",
        "ocd",
        "ocs",
        "phh2o",
        "sand",
        "silt",
        "soc",
    ]
    soil_depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    for measure in soil_measurements:
        for depth in soil_depths:
            khaki_columns.append(f"{measure}_mean_{depth}")

    # Add practice columns (P_1 to P_14) at the end
    for p in range(1, 15):
        khaki_columns.append(f"P_{p}")

    # Select only the columns that exist in our data
    available_columns = [col for col in khaki_columns if col in merged_df.columns]
    missing_columns = [col for col in khaki_columns if col not in merged_df.columns]

    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")

    # Create final dataframe
    khaki_df = merged_df[available_columns].copy()

    # Sort by loc_ID and year
    khaki_df = khaki_df.sort_values(["loc_ID", "year"]).reset_index(drop=True)  # type: ignore

    # Set output file path
    if output_file is None:
        output_file = f"khaki_{country}_multi_crop.csv"

    # Save to CSV
    logger.info(f"Saving Khaki format data to {output_file}")
    khaki_df.to_csv(output_file, index=False)

    logger.info(f"Conversion complete!")
    logger.info(f"Final shape: {khaki_df.shape}")
    logger.info(f"Columns: {list(khaki_df.columns)}")

    return output_file


def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert processed data to Khaki format"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Path to data directory"
    )
    parser.add_argument(
        "--country", type=str, required=True, help="Country name (e.g., argentina)"
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        required=True,
        help="Crop types (e.g., soybean corn wheat sunflower)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (optional)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set logging level based on command line argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        output_file = convert_to_khaki_format(
            data_dir=args.data_dir,
            country=args.country,
            crops=args.crops,
            output_file=args.output,
        )
        logger.info(f"Successfully created Khaki format file: {output_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
