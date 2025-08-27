#!/usr/bin/env python3
"""Process weather data using the new improved architecture"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.weather.config import WeatherConfig
from src.processing.weather.processor import WeatherProcessor
from src.downloader.weather.models import WeatherVariable
from src.constants import (
    DEFAULT_ADMIN_LEVEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_START_YEAR,
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_PARQUET,
)


def main():
    """Main function to process weather data using new architecture"""
    parser = argparse.ArgumentParser(
        description="Process weather data to county-level weekly averages (New Architecture)"
    )

    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country to process (e.g., 'Argentina', 'USA', 'BRA')",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"Start year (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="End year (default: current year)",
    )
    parser.add_argument(
        "--admin-level",
        type=int,
        default=DEFAULT_ADMIN_LEVEL,
        help=f"Admin level (1=state/province, 2=county/department, default: {DEFAULT_ADMIN_LEVEL})",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=[var.key for var in WeatherVariable],
        help="Variables to process (default: all)",
    )
    parser.add_argument(
        "--output-format",
        choices=[OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_PARQUET],
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output file format (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Create configuration
    config = WeatherConfig(
        country=args.country,
        start_year=args.start_year,
        end_year=args.end_year,
        variables=args.variables,
        admin_level=args.admin_level,
        data_dir=None,  # Will use default "data" directory
        output_format=args.output_format,
        debug=args.debug,
    )

    # Create and run processor
    processor = WeatherProcessor(config)

    try:
        output_paths = processor.process()
        print(f"\nProcessing complete! Generated {len(output_paths)} output files:")
        for path in output_paths:
            print(f"  {path}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
