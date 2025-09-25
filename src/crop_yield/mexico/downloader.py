"""
Mexico crop yield data downloader.

Downloads crop yield data from SIAP (Sistema de Información Agroalimentaria y Pesquera)
and saves raw HTML data to files. Supports downloading multiple years with parallel requests.
"""

import requests
import time
from pathlib import Path
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple

from src.crop_yield.mexico.models import (
    CROP_CODES,
    CROP_TYPE_CODE,
    API_BASE_URL,
    API_PARAMS,
)

logger = logging.getLogger(__name__)


def download_single_year(
    crop: str, crop_code: str, year: int, output_dir: Path
) -> Tuple[int, Path]:
    """
    Download Mexico crop yield data for a single year.

    Args:
        crop: Crop name (e.g., "corn")
        crop_code: SIAP crop code
        year: Year to download data for
        output_dir: Output directory for raw data

    Returns:
        Tuple of (year, output_file_path)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    output_file = output_dir / f"{crop}_{year}.html"
    if output_file.exists():
        logger.debug(f"File already exists, skipping download: {output_file}")
        return year, output_file

    # Build payload for API request using configuration from models
    payload = {
        "xajax": "reporte",
        "xajaxr": str(int(time.time() * 1000)),  # timestamp
        "xajaxargs[]": [
            API_PARAMS["nivel_municipio"],  # Position 0: 1
            str(year),  # Position 1: year (e.g., 2024)
            CROP_TYPE_CODE,  # Position 2: 5 (field crop type)
            API_PARAMS["modalidad_riego_temporal"],  # Position 3: 3 (Riego + Temporal)
            API_PARAMS["ciclo_todos"],  # Position 4: 0
            API_PARAMS["estado_placeholder"],  # Position 5: --
            API_PARAMS["municipio_placeholder"],  # Position 6: --
            crop_code,  # Position 7: crop code (225 for corn, 375 for soybean)
            API_PARAMS["municipio_code"],  # Position 8: 200201
            *API_PARAMS["additional_params"],  # Positions 9-13: 0,1,0,0,0
        ],
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # Create session for connection reuse
    session = requests.Session()

    # Send POST request
    response = session.post(API_BASE_URL, data=payload, headers=headers)
    response.raise_for_status()

    # Save raw HTML data by year
    with open(output_file, "w", encoding="iso-8859-1") as f:
        f.write(response.text)

    logger.debug(f"Downloaded {crop} data for {year} ({len(response.text)} chars)")
    return year, output_file


def download_mexico_crop_yield(
    crops: List[str],
    start_year: int,
    end_year: int,
    output_dir: Path,
    concurrent: int,
) -> List[Path]:
    """
    Download Mexico crop yield data for multiple crops and years with parallel requests.

    Args:
        crops: List of crop names (e.g., ["corn", "soybean"]) or empty list for all crops
        start_year: Starting year (default: 2003)
        end_year: Ending year (default: 2024)
        output_dir: Output directory for raw data
        concurrent: Maximum number of concurrent connections (default: 5)

    Returns:
        List of output file paths
    """
    # If no crops specified, use all available crops
    if not crops:
        crops = list(CROP_CODES.keys())
        print(
            f"No crops specified, downloading all available crops: {', '.join(crops)}"
        )

    # Validate all crop names - fail fast if any invalid
    for crop in crops:
        if crop not in CROP_CODES:
            raise ValueError(
                f"Unknown crop: {crop}. Available crops: {list(CROP_CODES.keys())}"
            )

    print(f"Downloading {len(crops)} crops: {', '.join(crops)}")

    # Generate list of years (end_year is exclusive)
    years = list(range(start_year, end_year))
    total_downloads = len(crops) * len(years)
    print(f"Downloading data for {len(years)} years ({start_year}-{end_year-1})")
    print(f"Total downloads: {total_downloads}")

    # Download data in parallel for all crop-year combinations
    output_files = []
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        # Submit all download tasks for all crops and years
        future_to_info = {}
        for crop in crops:
            crop_code = CROP_CODES[crop]
            for year in years:
                future = executor.submit(
                    download_single_year, crop, crop_code, year, output_dir
                )
                future_to_info[future] = (crop, year)

        # Process completed downloads with progress bar
        with tqdm(total=total_downloads, desc="Downloading crop data") as pbar:
            for future in as_completed(future_to_info):
                crop, year = future_to_info[future]
                try:
                    year_result, output_file = future.result()
                    output_files.append(output_file)
                    pbar.set_postfix(crop=crop, year=year)
                    pbar.update(1)
                except Exception:
                    pbar.update(1)
                    raise  # Fail fast - don't suppress errors

    print(f"Successfully downloaded {len(output_files)} files for {len(crops)} crops")
    return output_files


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Download Mexico crop yield data")
    parser.add_argument(
        "--crops",
        nargs="*",
        help="Crop names (e.g., corn soybean). If not specified, downloads all available crops",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1980,
        help="Starting year (inclusive, default: 1980)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Ending year (exclusive, default: 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mexico/raw/crop_yield"),
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent connections (default: 5)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Default to empty list if no crops specified (will download all)
    crops = args.crops or []

    try:
        output_files = download_mexico_crop_yield(
            crops, args.start_year, args.end_year, args.output_dir, args.concurrent
        )
        print(f"Successfully downloaded {len(output_files)} files")
        print(f"Raw data saved to: {args.output_dir}")
    except Exception:
        # Fail fast - let the exception propagate
        raise


if __name__ == "__main__":
    main()
