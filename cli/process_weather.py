#!/usr/bin/env python3
"""Process weather data using the new improved architecture"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_process_cli import (
    run_processor_cli,
    add_variables_argument,
    parse_variables,
)
from src.constants import DEFAULT_START_YEAR
from datetime import datetime
from src.processing.weather.config import WeatherConfig
from src.processing.weather.processor import WeatherProcessor
from src.downloader.weather.models import WeatherVariable


def add_weather_arguments(parser):
    """Add weather-specific arguments"""
    add_variables_argument(parser, WeatherVariable)
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


def parse_weather_arguments(args):
    """Parse weather-specific arguments and return config kwargs"""
    variables = None
    if args.variables:
        enum_vars = parse_variables(args.variables, WeatherVariable)
        if enum_vars:
            variables = [var.key for var in enum_vars]  # type: ignore

    return {
        "variables": variables,
        "start_year": args.start_year,
        "end_year": args.end_year,
    }


def main():
    """Main function to process weather data using new architecture"""
    run_processor_cli(
        description="Process weather data to county-level weekly averages (New Architecture)",
        config_class=WeatherConfig,
        processor_class=WeatherProcessor,
        add_custom_args_func=add_weather_arguments,
        parse_custom_args_func=parse_weather_arguments,
        success_message="Weather processing completed successfully!",
    )


if __name__ == "__main__":
    main()
