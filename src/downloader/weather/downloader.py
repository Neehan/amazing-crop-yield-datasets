"""Weather data downloader"""

import asyncio
import cdsapi
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.downloader.base import BaseDownloader
from src.downloader.weather.models import DownloadConfig, WeatherVariable

# Set up logger
logger = logging.getLogger(__name__)


class WeatherDownloader(BaseDownloader):
    """Downloads weather data from AgERA5 climate dataset via CDS API"""

    def __init__(self, data_dir: str, country: str, max_concurrent: int):
        """Initialize weather downloader with CDS API client

        Args:
            data_dir: Base directory to save downloaded files
            country: Country name for subdirectory
            max_concurrent: Maximum concurrent downloads
        """
        super().__init__(data_dir, country, "weather", max_concurrent)
        self.config = DownloadConfig()
        self.client = cdsapi.Client()

    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build CDS API request parameters"""
        year = kwargs["year"]
        variable = kwargs["variable"]
        bounds = kwargs.get("bounds")

        request = {
            "variable": variable.variable,
            "year": [str(year)],
            "month": [f"{month:02d}" for month in range(1, 13)],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "version": self.config.version,
        }

        if variable.statistic is not None:
            request["statistic"] = [variable.statistic]

        if bounds:
            request["area"] = bounds.to_cds_area()

        return {"dataset_name": self.config.dataset_name, "request": request}

    def _make_api_request(self, **kwargs) -> Any:
        """Make CDS API request"""
        return self.client.retrieve(kwargs["dataset_name"], kwargs["request"])

    async def _handle_api_response(self, result: Any, **kwargs):
        """Handle CDS API response by downloading the file"""
        file_path = kwargs.get("file_path")
        if file_path:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, result.download, str(file_path))

    async def download_weather(
        self,
        start_year: int,
        end_year: Optional[int],
        variables: Optional[List[WeatherVariable]],
    ):
        """Download AgERA5 weather data with concurrent processing

        Args:
            start_year: First year to download (inclusive)
            end_year: Last year to download (exclusive, default: current year)
            variables: Weather variables to download (default: all available)
        """
        end_year = end_year or datetime.now().year
        variables = variables or list(WeatherVariable)

        # Build download tasks directly
        bounds = self.get_country_bounds()
        download_tasks = []
        for variable in variables:
            for year in range(start_year, end_year):
                file_path = self.subdir / f"{year}_{variable.key}.zip"

                if not file_path.exists():
                    download_tasks.append(
                        {
                            "year": year,
                            "variable": variable,
                            "bounds": bounds,
                            "file_path": file_path,
                            "item_id": f"{variable.key}_{year}",
                        }
                    )

        if not download_tasks:
            print("All files already exist, skipping download")
            return

        print(f"Downloading {len(download_tasks)} files for {self.country}")
        await self.download(download_tasks, f"Weather data for {self.country}")
