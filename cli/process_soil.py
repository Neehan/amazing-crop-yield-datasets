#!/usr/bin/env python3
"""Process soil data from TIF files to aggregated CSV format"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_process_cli import run_processor_cli
from src.processing.soil.config import SoilConfig
from src.processing.soil.processor import SoilProcessor
from src.downloader.soil.models import SoilProperty, SoilDepth


def add_soil_arguments(parser):
    """Add soil-specific arguments"""
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=[prop.key for prop in SoilProperty],
        help="Soil properties to process (default: all)",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        choices=[depth.key for depth in SoilDepth],
        help="Depth ranges to process (default: all)",
    )


def parse_soil_arguments(args):
    """Parse soil-specific arguments and return config kwargs"""
    # Default to all properties if none specified
    properties = (
        args.properties if args.properties else [prop.key for prop in SoilProperty]
    )
    # Default to all depths if none specified
    depths = args.depths if args.depths else [depth.key for depth in SoilDepth]

    return {
        "properties": properties,
        "depths": depths,
    }


def main():
    run_processor_cli(
        description="Process soil data from TIF to aggregated CSV format",
        config_class=SoilConfig,
        processor_class=SoilProcessor,
        add_custom_args_func=add_soil_arguments,
        parse_custom_args_func=parse_soil_arguments,
        success_message="Soil processing completed successfully!",
    )


if __name__ == "__main__":
    main()
