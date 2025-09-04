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
        required=True,
        help="Soil properties to process",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        choices=[depth.key for depth in SoilDepth],
        required=True,
        help="Depth ranges to process",
    )


def parse_soil_arguments(args):
    """Parse soil-specific arguments and return config kwargs"""
    return {
        "properties": args.properties,
        "depths": args.depths,
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
