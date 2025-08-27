#!/usr/bin/env python3
"""Download AgERA5 weather data globally"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_download_cli import run_downloader_cli
from src.downloader.weather import WeatherDownloader, WeatherVariable
from src.constants import DATA_DIR


def main():
    """Main function to download weather data"""
    run_downloader_cli(
        description="Download AgERA5 weather data",
        variable_enum=WeatherVariable,
        downloader_class=WeatherDownloader,
        download_method="download_weather",
        data_dir=DATA_DIR,
    )


if __name__ == "__main__":
    main()
