#!/usr/bin/env python3
"""Process irrigation fraction data using HYDE dataset"""

from cli.base_process_cli import run_processor_cli
from src.processing.management.irrigation.config import IrrigationConfig
from src.processing.management.irrigation.processor import IrrigationProcessor
from datetime import datetime


def add_irrigation_arguments(parser):
    """Add irrigation-specific arguments"""
    parser.add_argument(
        "--start-year",
        type=int,
        default=1979,
        help="Start year (default: 2000)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="End year (default: 2000)",
    )


def parse_irrigation_arguments(args):
    """Parse irrigation-specific arguments and return config kwargs"""
    return {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "variables": None,  # Not used for irrigation
    }


def main():
    """Main function to process irrigation fraction data"""
    run_processor_cli(
        description="Process irrigation fraction data from HYDE dataset",
        config_class=IrrigationConfig,
        processor_class=IrrigationProcessor,
        add_custom_args_func=add_irrigation_arguments,
        parse_custom_args_func=parse_irrigation_arguments,
        success_message="Irrigation fraction processing completed successfully!",
    )


if __name__ == "__main__":
    main()
