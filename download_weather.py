#!/usr/bin/env python3
"""Download AgERA5 weather data globally"""

import argparse
import logging
from datetime import datetime

from src.weather import download_weather, WeatherVariable


def main():
    """Main function to download weather data"""
    parser = argparse.ArgumentParser(description="Download AgERA5 weather data")

    parser.add_argument("--start-year", type=int, default=1979, help="Start year")
    parser.add_argument(
        "--end-year", type=int, default=datetime.now().year - 1, help="End year"
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=[var.key for var in WeatherVariable],
        help="Variables to download (default: all)",
    )
    parser.add_argument(
        "--list-variables", action="store_true", help="List available variables"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Number of concurrent downloads (default: 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows CDS API requests)",
    )

    args = parser.parse_args()

    # Set up logging after parsing args
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.list_variables:
        print("Available weather variables:")
        for var in WeatherVariable:
            print(f"  {var.key}: {var.variable} ({var.statistic})")
        return

    # Convert variable keys to enums if specified
    variables = None
    if args.variables:
        variables = [var for var in WeatherVariable if var.key in args.variables]

    download_weather(
        start_year=args.start_year,
        end_year=args.end_year,
        variables=variables,
        max_concurrent=args.concurrent,
    )


if __name__ == "__main__":
    main()
