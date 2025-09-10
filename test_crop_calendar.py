#!/usr/bin/env python3
"""Test script for crop calendar processing pipeline"""

import argparse
from pathlib import Path

from src.processing.crop_calendar.config import CropCalendarConfig
from src.processing.crop_calendar.processor import CropCalendarProcessor

def main():
    """Test crop calendar processing"""
    parser = argparse.ArgumentParser(description="Test crop calendar processing")
    parser.add_argument("--country", default="Argentina", help="Country to test")
    parser.add_argument("--crops", nargs="+", default=["wheat"], help="Crop names to test (default: wheat)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Create config
    config = CropCalendarConfig(
        country=args.country,
        crop_names=args.crops,
        admin_level=2,
        data_dir=Path("data"),
        output_format="csv", 
        debug=args.debug
    )
    
    # Process
    processor = CropCalendarProcessor(config)
    output_files = processor.process_with_validation()
    
    print(f"\nProcessing complete! Generated {len(output_files)} files:")
    for file_path in output_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()