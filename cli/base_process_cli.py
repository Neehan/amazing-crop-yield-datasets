#!/usr/bin/env python3
"""Base CLI functionality for data processors"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import (
    DEFAULT_ADMIN_LEVEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_START_YEAR,
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_PARQUET,
)


def create_base_process_parser(description: str) -> argparse.ArgumentParser:
    """Create base argument parser with common processing options"""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country to process (e.g., 'Argentina', 'USA', 'Brazil')",
    )
    parser.add_argument(
        "--admin-level",
        type=int,
        default=DEFAULT_ADMIN_LEVEL,
        help=f"Admin level (0=country, 1=state/province, 2=county/department, default: {DEFAULT_ADMIN_LEVEL})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Base data directory (default: ./data)",
    )
    parser.add_argument(
        "--output-format",
        choices=[OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_PARQUET],
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output file format (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser


def setup_logging(debug: bool):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def add_variables_argument(parser: argparse.ArgumentParser, variable_enum: Any):
    """Add variables argument with choices from the given enum"""
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=[var.key for var in variable_enum],
        help="Variables to process (default: all)",
    )


def parse_variables(
    variable_keys: Optional[List[str]], variable_enum: Any
) -> Optional[List[Any]]:
    """Convert variable keys to enum instances"""
    if not variable_keys:
        return None
    return [var for var in variable_enum if var.key in variable_keys]


def run_processor_cli(
    description: str,
    config_class: Any,
    processor_class: Any,
    add_custom_args_func: Optional[Any] = None,
    parse_custom_args_func: Optional[Any] = None,
    success_message: str = "Processing completed successfully!",
):
    """Run a processor CLI with flexible functionality

    Args:
        description: CLI description
        config_class: Configuration class to instantiate
        processor_class: Processor class to instantiate
        add_custom_args_func: Function to add custom arguments to parser
        parse_custom_args_func: Function to parse custom arguments and return config kwargs
        success_message: Message to display on successful completion
    """
    parser = create_base_process_parser(description)

    # Add custom arguments if function provided
    if add_custom_args_func:
        add_custom_args_func(parser)

    args = parser.parse_args()

    setup_logging(args.debug)

    try:
        # Base configuration
        config_kwargs = {
            "country": args.country,
            "admin_level": args.admin_level,
            "output_format": args.output_format,
            "debug": args.debug,
            "data_dir": args.data_dir,
        }

        # Parse custom arguments if function provided
        if parse_custom_args_func:
            custom_kwargs = parse_custom_args_func(args)
            config_kwargs.update(custom_kwargs)

        config = config_class(**config_kwargs)

        # Create and run processor
        processor = processor_class(config)
        output_files = processor.process()

        logger = logging.getLogger(__name__)
        logger.info(success_message)
        logger.info(f"Generated {len(output_files)} output files:")
        for file_path in output_files:
            logger.info(f"  {file_path}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Processing failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)
