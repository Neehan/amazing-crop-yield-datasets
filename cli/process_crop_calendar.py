#!/usr/bin/env python3
"""Process crop calendar data using the new weighted architecture"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_process_cli import run_processor_cli
from src.processing.crop_calendar.config import CropCalendarConfig
from src.processing.crop_calendar.processor import CropCalendarProcessor


def add_crop_calendar_arguments(parser):
    """Add crop calendar-specific arguments"""
    parser.add_argument(
        "--crops",
        nargs="*",
        default=None,
        help="Crop names to process (if not specified, auto-detect from final directory)",
    )


def parse_crop_calendar_arguments(args):
    """Parse crop calendar-specific arguments and return config kwargs"""
    return {
        "crop_names": args.crops,
    }


def main():
    """Main function to process crop calendar data using new weighted architecture"""
    run_processor_cli(
        description="Process crop calendar data to admin-level weighted averages (New Architecture)",
        config_class=CropCalendarConfig,
        processor_class=CropCalendarProcessor,
        add_custom_args_func=add_crop_calendar_arguments,
        parse_custom_args_func=parse_crop_calendar_arguments,
        success_message="Crop calendar processing completed successfully!",
    )


if __name__ == "__main__":
    main()
