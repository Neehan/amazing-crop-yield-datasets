#!/usr/bin/env python3
"""Process weather data to county-level weekly averages"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.processing.base_processor import BaseProcessor
from src.processing.spatial_aggregator import SpatialAggregator
from src.processing.temporal_aggregator import TemporalAggregator
from src.processing.zip_extractor import ZipExtractor
from src.weather.models import WeatherVariable


def main():
    """Main function to process weather data"""
    parser = argparse.ArgumentParser(
        description="Process weather data to county-level weekly averages"
    )

    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country to process (e.g., 'Argentina', 'USA', 'BRA')",
    )
    parser.add_argument("--start-year", type=int, default=2020, help="Start year")
    parser.add_argument(
        "--end-year", type=int, default=datetime.now().year, help="End year"
    )
    parser.add_argument(
        "--admin-level",
        type=int,
        default=2,
        help="Admin level (1=state/province, 2=county/department)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=[var.key for var in WeatherVariable],
        help="Variables to process (default: all)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Processing weather data for {args.country}")

    # Set up data directories
    data_dir = Path("data")
    weather_dir = data_dir / args.country.lower() / "weather"

    # Check input data exists
    if not weather_dir.exists():
        logger.error(f"Weather data directory not found: {weather_dir}")
        logger.info(f"Please download weather data first:")
        logger.info(
            f"python download_weather.py --country {args.country} --start-year {args.start_year} --end-year {args.end_year}"
        )
        return

    # Find input zip files
    zip_files = list(weather_dir.glob("*.zip"))
    if not zip_files:
        logger.error(f"No .zip files found in {weather_dir}")
        return

    logger.debug(f"Found {len(zip_files)} zip files to process")

    # Determine variables to process
    variables_to_process = args.variables if args.variables else [var.key for var in WeatherVariable]
    logger.info(f"Processing variables: {variables_to_process}")

    # Save to processed directory directly (no weather subfolder)
    processed_dir = data_dir / args.country.lower() / "processed"

    # Initialize processors for spatial aggregation
    logger.info(
        f"Loading administrative boundaries for {args.country} at level {args.admin_level}"
    )
    base_processor = BaseProcessor(args.country, args.admin_level)
    logger.debug(f"Loaded {len(base_processor.boundaries)} administrative units")

    spatial_aggregator = SpatialAggregator(base_processor, args.country)
    temporal_aggregator = TemporalAggregator()
    zip_extractor = ZipExtractor()

    # Process each variable separately
    for variable in variables_to_process:
        logger.info(f"Processing variable: {variable}")

        # Step 1: Extract and combine all years for this variable into single NetCDF file
        logger.info(f"Extracting and combining all years for {variable}...")
        combined_nc_file = zip_extractor.process_all_zips_to_single_file(
            weather_dir=weather_dir,
            output_dir=processed_dir,
            year_range=(args.start_year, args.end_year),
            variables=[variable],
            force_refresh=False,
        )

        logger.debug(f"Created combined NetCDF file for {variable}: {combined_nc_file}")

        # Step 2: Spatial aggregation for this variable
        logger.info(f"Processing spatial aggregation for {variable}...")
        combined_spatial = spatial_aggregator.aggregate_file(
            str(combined_nc_file)
        )

        # Add variable name to the result
        combined_spatial["variable"] = variable
        logger.debug(f"Spatial aggregation complete for {variable}: {combined_spatial.shape}")

        # Step 3: Temporal aggregation to weekly averages and pivot format
        logger.info(f"Converting {variable} to weekly averages and pivot format...")
        weekly_pivoted = temporal_aggregator.daily_to_weekly_pivot(
            combined_spatial, admin_level=args.admin_level
        )
        logger.debug(f"Weekly pivot data shape for {variable}: {weekly_pivoted.shape}")

        # Missing data analysis
        total_cells = weekly_pivoted.shape[0] * weekly_pivoted.shape[1]
        missing_cells = weekly_pivoted.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        # Column-wise missing data
        missing_by_col = weekly_pivoted.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0]
        
        logger.info(f"Missing data analysis for {variable}:")
        logger.info(f"  Total missing cells: {missing_cells:,} / {total_cells:,} ({missing_percentage:.2f}%)")
        if len(cols_with_missing) > 0:
            logger.info(f"  Columns with missing data: {len(cols_with_missing)}")
            for col, count in cols_with_missing.head(10).items():
                logger.info(f"    {col}: {count} missing")
        else:
            logger.info("  No missing data found")

        # Save each variable separately
        output_filename = f"weather_{args.start_year}-{args.end_year-1}_{variable}_weekly_weighted_admin{args.admin_level}.csv"
        output_path = processed_dir / output_filename
        weekly_pivoted.to_csv(output_path, index=False)
        
        logger.info(f"Saved {variable} to: {output_path}")
        logger.info(f"Dataset shape: {weekly_pivoted.shape[0]} rows, {weekly_pivoted.shape[1]} columns")

    logger.info(f"Processing complete! All {len(variables_to_process)} variables saved to {processed_dir}")


if __name__ == "__main__":
    main()
