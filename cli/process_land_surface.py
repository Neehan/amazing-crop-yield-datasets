#!/usr/bin/env python3
"""CLI for processing land surface data"""

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
from src.processing.land_surface.config import LandSurfaceConfig
from src.processing.land_surface.processor import LandSurfaceProcessor
from src.downloader.land_surface.models import LandSurfaceVariable


def add_land_surface_arguments(parser):
    """Add land surface-specific arguments"""
    add_variables_argument(parser, LandSurfaceVariable)
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


def parse_land_surface_arguments(args):
    """Parse land surface-specific arguments and return config kwargs"""
    return {
        "variables": parse_variables(
            getattr(args, "variables", None), LandSurfaceVariable
        ),
        "start_year": args.start_year,
        "end_year": args.end_year,
    }


def main():
    run_processor_cli(
        description="Process land surface data (LAI, NDVI, etc.) to CSV format aggregated by administrative boundaries",
        config_class=LandSurfaceConfig,
        processor_class=LandSurfaceProcessor,
        add_custom_args_func=add_land_surface_arguments,
        parse_custom_args_func=parse_land_surface_arguments,
        success_message="Land surface processing completed successfully!",
    )


if __name__ == "__main__":
    main()
