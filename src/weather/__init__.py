"""Weather data downloading module"""

from typing import List, Optional
from src.weather.downloader import WeatherDownloader
from src.weather.models import WeatherVariable


# Clean API
def download_weather(
    start_year: int = 1979,
    end_year: Optional[int] = None,
    variables: Optional[List[WeatherVariable]] = None,
    max_concurrent: int = 3,
):
    """Download global AgERA5 weather data with year-based iteration

    This is the main API function. Downloads global data year by year for each variable
    with transparent error reporting.

    Args:
        start_year: First year to download (default: 1979)
        end_year: Last year to download (default: previous year)
        variables: Weather variables to download (default: all available)
        max_concurrent: Maximum concurrent downloads (not used in current implementation)

    Example:
        # Download all data for 2020-2023
        download_weather(2020, 2023)

        # Download only temperature
        download_weather(variables=[WeatherVariable.TEMP_MIN, WeatherVariable.TEMP_MAX])
    """
    downloader = WeatherDownloader(max_concurrent=max_concurrent)
    downloader.download(start_year, end_year, variables)


__all__ = ["download_weather", "WeatherDownloader", "WeatherVariable"]
