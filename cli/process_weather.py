#!/usr/bin/env python3
"""Process weather data using the new improved architecture"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_process_cli import run_processor_cli
from src.processing.weather.config import WeatherConfig
from src.processing.weather.processor import WeatherProcessor
from src.downloader.weather.models import WeatherVariable


def main():
    """Main function to process weather data using new architecture"""
    run_processor_cli(
        description="Process weather data to county-level weekly averages (New Architecture)",
        config_class=WeatherConfig,
        processor_class=WeatherProcessor,
        variable_enum=WeatherVariable,
        success_message="Weather processing completed successfully!",
    )


if __name__ == "__main__":
    main()
