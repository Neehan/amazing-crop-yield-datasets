"""Weather data downloading module"""

import asyncio
from typing import List, Optional
from src.downloader.weather.downloader import WeatherDownloader
from src.downloader.weather.models import WeatherVariable
from src.constants import DATA_DIR


def download_weather(
    start_year: int,
    end_year: Optional[int],
    variables: Optional[List[WeatherVariable]],
    max_concurrent: int,
    country: str,
):
    """Download AgERA5 weather data with concurrent processing

    Args:
        start_year: First year to download (inclusive, default: 1979)
        end_year: Last year to download (exclusive, default: current year)
        variables: Weather variables to download (default: all available)
        max_concurrent: Maximum concurrent downloads (default: 5)
        country: Country name to filter data (default: None for global data)
    """
    if not country:
        raise ValueError("Country must be specified")

    downloader = WeatherDownloader(str(DATA_DIR), country, max_concurrent)
    asyncio.run(downloader.download_weather(start_year, end_year, variables))


__all__ = ["download_weather", "WeatherDownloader", "WeatherVariable"]
