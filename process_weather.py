#!/usr/bin/env python3
"""Process weather data to county-level weekly averages"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
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
    parser.add_argument(
        "--spatial-method",
        type=str,
        default="mean",
        choices=["mean", "sum", "median"],
        help="Spatial aggregation method",
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

        # Step 1: Extract and combine all years into single NetCDF file
    logger.info("Extracting and combining all years into single NetCDF file...")
    zip_extractor = ZipExtractor()

    # Save to processed directory directly (no weather subfolder)
    processed_dir = data_dir / args.country.lower() / "processed"

    combined_nc_file = zip_extractor.process_all_zips_to_single_file(
        weather_dir=weather_dir,
        output_dir=processed_dir,
        year_range=(args.start_year, args.end_year),
        variables=args.variables,
        force_refresh=False,
    )

    logger.info(f"Created combined multi-year NetCDF file: {combined_nc_file}")

    # Step 2: Initialize processors for spatial aggregation
    logger.info(
        f"Loading administrative boundaries for {args.country} at level {args.admin_level}"
    )
    base_processor = BaseProcessor(args.country, args.admin_level)
    logger.debug(f"Loaded {len(base_processor.boundaries)} administrative units")

    spatial_aggregator = SpatialAggregator(base_processor)
    temporal_aggregator = TemporalAggregator()

    # Step 3: Single spatial aggregation for all years at once
    logger.info("Processing spatial aggregation for all years at once...")
    combined_spatial = spatial_aggregator.aggregate_file(
        str(combined_nc_file), method=args.spatial_method
    )

    # Add variable name to the result
    if args.variables:
        combined_spatial["variable"] = (
            args.variables[0] if len(args.variables) == 1 else "multi_var"
        )
    else:
        combined_spatial["variable"] = "all_vars"

    logger.debug(f"Spatial aggregation complete: {combined_spatial.shape}")

    # Step 4: Temporal aggregation to weekly averages and pivot format
    logger.info("Converting to weekly averages and pivot format...")
    weekly_pivoted = temporal_aggregator.daily_to_weekly_pivot(
        combined_spatial, admin_level=args.admin_level
    )
    logger.debug(f"Weekly pivot data shape: {weekly_pivoted.shape}")

    # Generate output filename (no country name in filename)
    var_suffix = "_".join(args.variables) if args.variables else "all-vars"
    output_filename = f"weather_{args.start_year}-{args.end_year-1}_{var_suffix}_weekly_{args.spatial_method}_admin{args.admin_level}.csv"
    output_path = processed_dir / output_filename

    # Save results
    weekly_pivoted.to_csv(output_path, index=False)

    logger.info(f"Processing complete! Output saved to: {output_path}")
    logger.info(
        f"Final dataset: {weekly_pivoted.shape[0]} rows, {weekly_pivoted.shape[1]} columns"
    )

    # Show sample of first few columns
    sample_cols = list(weekly_pivoted.columns)[:8]
    logger.debug(f"Sample columns: {sample_cols}")


if __name__ == "__main__":
    main()
