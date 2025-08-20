"""Weather data downloading module"""

import asyncio
from typing import List, Optional
from src.weather.downloader import WeatherDownloader
from src.weather.models import WeatherVariable


# Clean API
def download_weather(
    start_year: int = 1979,
    end_year: Optional[int] = None,
    variables: Optional[List[WeatherVariable]] = None,
    max_concurrent: int = 5,
    country: Optional[str] = None,
):
    """Download global AgERA5 weather data with concurrent year processing

    This is the main API function. Downloads data with concurrent year processing
    for improved performance.

    Args:
        start_year: First year to download (inclusive, default: 1979)
        end_year: Last year to download (exclusive, default: current year)
        variables: Weather variables to download (default: all available)
        max_concurrent: Maximum concurrent downloads (default: 5)
        country: Country name to filter data (default: None for global data)

    Example:
        # Download all data for 2020-2023
        download_weather(2020, 2023)

        # Download only temperature with 6 concurrent downloads
        download_weather(variables=[WeatherVariable.T2M_MIN, WeatherVariable.T2M_MAX], max_concurrent=6)
    """
    downloader = WeatherDownloader(max_concurrent=max_concurrent)
    asyncio.run(downloader.download(start_year, end_year, variables, country))


__all__ = ["download_weather", "WeatherDownloader", "WeatherVariable"]
