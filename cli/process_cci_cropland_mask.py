#!/usr/bin/env python3
"""Process CCI cropland mask data to create cropland and irrigated masks"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.management.cci_cropland_mask import (
    CCICroplandMaskProcessor,
    CCICroplandMaskConfig,
)
from src.constants import DATA_DIR


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process CCI cropland mask data to create cropland and irrigated masks"
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country name (e.g., 'Argentina', 'Brazil', 'USA')",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start year for processing (default: 2020)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2022,
        help="End year for processing (default: 2022)",
    )
    parser.add_argument(
        "--admin-level",
        type=int,
        default=2,
        help="Administrative level for processing (default: 2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Base data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.debug)

    logger = logging.getLogger(__name__)

    try:
        # Create configuration
        config = CCICroplandMaskConfig(
            country=args.country,
            start_year=args.start_year,
            end_year=args.end_year,
            admin_level=args.admin_level,
            data_dir=Path(args.data_dir),
            output_format=args.output_format,
            debug=args.debug,
        )

        # Create processor and run
        processor = CCICroplandMaskProcessor(config)
        output_files = processor.process_with_validation()

        logger.info(
            f"Successfully processed {len(output_files)} CCI cropland mask data."
        )
        for filepath in output_files:
            logger.debug(f"  - {filepath}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
