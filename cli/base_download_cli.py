#!/usr/bin/env python3
"""Base CLI functionality for data downloaders"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create base argument parser with common options"""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--start-year", type=int, default=1979, help="Start year (inclusive)"
    )
    parser.add_argument(
        "--end-year", type=int, default=datetime.now().year, help="End year (exclusive)"
    )
    parser.add_argument(
        "--list-variables", action="store_true", help="List available variables"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Number of concurrent downloads (default: 5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows CDS API requests)",
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Country name to filter data (e.g., 'USA', 'Brazil', 'Argentina'). Required for downloading.",
    )

    return parser


def setup_logging(debug: bool):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def list_variables(variable_enum: Any):
    """List available variables for the given enum"""
    print(
        f"Available {variable_enum.__name__.replace('Variable', '').lower()} variables:"
    )
    for var in variable_enum:
        print(f"  {var.key}: {var.variable} ({var.statistic})")


def validate_country(country: Optional[str]):
    """Validate that country is provided"""
    if not country:
        print("Error: --country is required for downloading data")
        sys.exit(1)


def parse_variables(
    variable_keys: Optional[List[str]], variable_enum: Any
) -> Optional[List[Any]]:
    """Convert variable keys to enum instances"""
    if not variable_keys:
        return None
    return [var for var in variable_enum if var.key in variable_keys]


def run_downloader_cli(
    description: str,
    variable_enum: Any,
    downloader_class: Any,
    download_method: str,
    data_dir: str,
):
    """Run a downloader CLI with common functionality

    Args:
        description: CLI description
        variable_enum: Enum class for variables
        downloader_class: Downloader class to instantiate
        download_method: Name of the download method to call
        data_dir: Base data directory
    """
    import asyncio

    parser = create_base_parser(description)

    # Add variables argument with choices from the enum
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=[var.key for var in variable_enum],
        help="Variables to download (default: all)",
    )

    args = parser.parse_args()

    setup_logging(args.debug)

    if args.list_variables:
        list_variables(variable_enum)
        return

    validate_country(args.country)

    variables = parse_variables(args.variables, variable_enum)

    downloader = downloader_class(data_dir, args.country, args.concurrent)
    download_func = getattr(downloader, download_method)
    asyncio.run(download_func(args.start_year, args.end_year, variables))
