#!/usr/bin/env python3
"""Download SoilGrids soil data globally"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.soil import SoilDownloader, SoilProperty, SoilDepth
from src.constants import DATA_DIR


def setup_logging(debug: bool):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def list_properties():
    """List available soil properties"""
    logging.info("Available soil properties:")
    for prop in SoilProperty:
        logging.info(f"  * {prop.key}: {prop.code} - {prop.description}")


def list_depths():
    """List available depth ranges"""
    logging.info("Available depth ranges:")
    for depth in SoilDepth:
        logging.info(
            f"  {depth.key}: {depth.range_str} ({depth.start_cm}-{depth.end_cm} cm)"
        )


def parse_properties(
    property_keys: Optional[List[str]],
) -> Optional[List[SoilProperty]]:
    """Convert property keys to enum instances"""
    if not property_keys:
        return None
    return [prop for prop in SoilProperty if prop.key in property_keys]


def parse_depths(depth_keys: Optional[List[str]]) -> Optional[List[SoilDepth]]:
    """Convert depth keys to enum instances"""
    if not depth_keys:
        return None
    return [depth for depth in SoilDepth if depth.key in depth_keys]


def main():
    """Main function to download soil data"""
    parser = argparse.ArgumentParser(description="Download SoilGrids soil data")

    all_soil_properties = [prop.key for prop in SoilProperty]
    all_soil_depths = [depth.key for depth in SoilDepth]

    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country name (e.g., 'USA', 'Brazil', 'Argentina')",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=all_soil_properties,
        help="Soil properties to download (default: all)",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        choices=all_soil_depths,
        help="Depth ranges to download (default: all available for each property)",
    )
    parser.add_argument(
        "--list-properties", action="store_true", help="List available soil properties"
    )
    parser.add_argument(
        "--list-depths", action="store_true", help="List available depth ranges"
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
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.debug)

    if args.list_properties:
        list_properties()
        return

    if args.list_depths:
        list_depths()
        return

    properties = parse_properties(args.properties)
    depths = parse_depths(args.depths)

    logging.info(f"Downloading soil data for {args.country}")
    if properties:
        properties_formatted = "\n * ".join([p.code for p in properties])
        logging.info(f"Properties: \n * {properties_formatted}")
    else:
        properties_formatted = "\n * ".join(all_soil_properties)
        logging.info(f"Properties: \n * {properties_formatted}")

    if depths:
        depths_formatted = "\n * ".join([d.range_str for d in depths])
        logging.info(f"Depths: \n * {depths_formatted}")
    else:
        depths_formatted = "\n * ".join(all_soil_depths)
        logging.info(f"Depths: \n * {depths_formatted}")

    downloader = SoilDownloader(DATA_DIR, args.country, args.concurrent)
    asyncio.run(downloader.download_soil(properties, depths))


if __name__ == "__main__":
    main()
