"""Weather data downloader"""

import cdsapi
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from tqdm import tqdm

from src.weather.models import WeatherVariable, DownloadConfig, GeoBounds
from src.weather.geography import Geography

# Set up logger
logger = logging.getLogger(__name__)


class WeatherDownloader:
    """Downloads weather data from AgERA5 climate dataset via CDS API"""

    def __init__(self, data_dir: str = "data", max_concurrent: int = 3):
        """Initialize downloader with data directory and API client

        Args:
            data_dir: Base directory to save downloaded files
            max_concurrent: Maximum concurrent downloads
        """
        self.data_dir = Path(data_dir)
        self.config = DownloadConfig()
        self.geography = Geography()
        self.max_concurrent = max_concurrent
        self.client = cdsapi.Client()

    def download(
        self,
        start_year: int = 1979,
        end_year: Optional[int] = None,
        variables: Optional[List[WeatherVariable]] = None,
    ):
        """Download global weather data

        Args:
            start_year: First year to download data for
            end_year: Last year to download data for (defaults to previous year)
            variables: List of weather variables to download (defaults to all)
        """
        end_year = end_year or datetime.now().year - 1
        variables = variables or list(WeatherVariable)

        output_dir = self._ensure_weather_directory_exists()

        logger.info(f"Downloading global weather data ({start_year}-{end_year})")

        # Download each variable for each year
        for variable in variables:
            logger.info(f"Downloading {variable.key}...")
            for year in tqdm(range(start_year, end_year), desc=f"{variable.key}"):
                file_path = output_dir / f"{year}_{variable.key}.nc"
                
                if file_path.exists():
                    logger.debug(f"Skipping existing file: {file_path}")
                    continue
                
                request = self._build_cds_request(year, variable)
                logger.debug(f"CDS API Request: {request}")
                result = self.client.retrieve(self.config.dataset_name, request)
                result.download(str(file_path))
                logger.info(f"Downloaded: {file_path}")

    def _ensure_weather_directory_exists(self) -> Path:
        """Create and return the global weather data directory

        Returns:
            Path to the weather data directory
        """
        weather_dir = self.data_dir / "weather"
        weather_dir.mkdir(parents=True, exist_ok=True)
        return weather_dir

    def _build_cds_request(
        self,
        year: int,
        variable: WeatherVariable,
    ) -> dict:
        """Build a request dictionary for the AgERA5 CDS API

        Args:
            year: Year of data to request
            variable: Weather variable to download

        Returns:
            Dictionary formatted for CDS API request
        """
        request = {
            "variable": variable.variable,
            "year": [str(year)],
            "month": [f"{month:02d}" for month in range(1, 13)],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "version": self.config.version
        }
        
        # Add statistic for variables that require it
        if variable.statistic:
            request["statistic"] = [variable.statistic]
            
        return request

