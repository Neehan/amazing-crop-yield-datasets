#!/usr/bin/env python3
"""Download ERA5 land surface data globally"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_download_cli import run_downloader_cli
from src.downloader.land_surface import LandSurfaceDownloader, LandSurfaceVariable
from src.constants import DATA_DIR


def main():
    """Main function to download land surface data"""
    run_downloader_cli(
        description="Download ERA5 land surface data",
        variable_enum=LandSurfaceVariable,
        downloader_class=LandSurfaceDownloader,
        download_method="download_land_surface",
        data_dir=DATA_DIR,
    )


if __name__ == "__main__":
    main()
