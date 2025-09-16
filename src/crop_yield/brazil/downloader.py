"""
Brazil crop yield data downloader.

Downloads crop yield data from IBGE PAM 1612 table and saves raw data to CSV files.
Supports downloading multiple years with parallel API calls.
"""

import sidrapy
import pandas as pd
from pathlib import Path
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple

from src.crop_yield.brazil.models import CROP_CODES

logger = logging.getLogger(__name__)


def download_single_year(
    crop: str, crop_code: str, year: int, output_dir: Path
) -> Tuple[int, Path]:
    """
    Download Brazil crop yield data for a single year.

    Args:
        crop: Crop name (e.g., "corn", "wheat", "soybean")
        crop_code: IBGE crop code
        year: Year to download data for
        output_dir: Output directory for raw data

    Returns:
        Tuple of (year, output_file_path)
    """
    try:
        # Query PAM 1612 for crop yield data
        data = sidrapy.get_table(
            table_code="1612",
            territorial_level="6",  # N6 = municipality
            ibge_territorial_code="all",  # all municipalities
            period=str(year),
            variable="112",  # rendimento médio (kg/ha)
            classification="81",  # C81 = produto lavouras temporárias
            categories=crop_code,  # specific crop code
        )

        df = pd.DataFrame(data)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw data by year
        output_file = output_dir / f"{crop}_{year}.csv"
        df.to_csv(output_file, index=False)

        logger.debug(f"Downloaded {len(df)} records for {crop} in {year}")
        return year, output_file

    except Exception as e:
        logger.error(f"Failed to download {crop} data for {year}: {e}")
        raise


def download_brazil_crop_yield(
    crop: str,
    start_year: int = 1974,
    end_year: int = 2024,
    output_dir: Path = Path("data/brazil/raw/crop_yield"),
    max_workers: int = 5,
) -> List[Path]:
    """
    Download Brazil crop yield data for multiple years with parallel API calls.

    Args:
        crop: Crop name (e.g., "corn", "wheat", "soybean")
        start_year: Starting year (default: 1974)
        end_year: Ending year (default: 2024)
        output_dir: Output directory for raw data
        max_workers: Maximum number of parallel workers (default: 5)

    Returns:
        List of output file paths
    """
    # Convert crop name to code if needed
    if crop in CROP_CODES:
        crop_code = CROP_CODES[crop]
        print(f"Using crop code {crop_code} for {crop}")
    else:
        raise ValueError(
            f"Invalid crop name: {crop}. Valid options: {CROP_CODES.keys()}"
        )

    # Generate list of years (end_year is exclusive)
    years = list(range(start_year, end_year))
    print(f"Downloading {crop} data for {len(years)} years ({start_year}-{end_year-1})")

    # Download data in parallel
    output_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_year = {
            executor.submit(
                download_single_year, crop, crop_code, year, output_dir
            ): year
            for year in years
        }

        # Process completed downloads with progress bar
        with tqdm(total=len(years), desc=f"Downloading {crop}") as pbar:
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_result, output_file = future.result()
                    output_files.append(output_file)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to download {crop} data for {year}: {e}")
                    pbar.update(1)

    print(f"Successfully downloaded {len(output_files)} years of {crop} data")
    return output_files


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Download Brazil crop yield data")
    parser.add_argument(
        "--crop", required=True, help="Crop name (e.g., corn, wheat, soybean)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1974,
        help="Starting year (inclusive, default: 1974)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Ending year (exclusive, default: 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/brazil/raw/crop_yield"),
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        output_files = download_brazil_crop_yield(
            args.crop, args.start_year, args.end_year, args.output_dir, args.max_workers
        )
        print(
            f"Successfully downloaded {len(output_files)} years of {args.crop} yield data"
        )
        print(f"Raw data saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise


if __name__ == "__main__":
    main()
