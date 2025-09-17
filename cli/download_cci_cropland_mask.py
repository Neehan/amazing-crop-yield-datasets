#!/usr/bin/env python3
"""Download CCI Cropland Mask data per country"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_download_cli import run_downloader_cli
from src.downloader.cci_cropland_mask import (
    CCICroplandMaskDownloader,
    CCICroplandMaskVariable,
)
from src.constants import DATA_DIR


def main():
    """Main function to download CCI cropland mask data"""
    run_downloader_cli(
        description="Download CCI Cropland Mask",
        variable_enum=CCICroplandMaskVariable,
        downloader_class=CCICroplandMaskDownloader,
        download_method="download_cci_cropland_mask",
        data_dir=DATA_DIR,
    )


if __name__ == "__main__":
    main()
