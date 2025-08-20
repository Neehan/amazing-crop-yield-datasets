"""Weather data downloader"""

import cdsapi
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from tqdm import tqdm

from src.weather.models import WeatherVariable, DownloadConfig
from src.utils.geography import Geography, GeoBounds

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
        country: Optional[str] = None,
    ):
        """Download weather data (global or country-specific)

        Args:
            start_year: First year to download data for
            end_year: Last year to download data for (defaults to previous year)
            variables: List of weather variables to download (defaults to all)
            country: Country name to filter data (defaults to global)
        """
        end_year = end_year or datetime.now().year - 1
        variables = variables or list(WeatherVariable)

        # Get country bounds if specified
        bounds = None
        if country:
            bounds = self.geography.get_country_bounds(country)
            logger.info(f"Downloading weather data for {country} ({start_year}-{end_year})")
        else:
            logger.info(f"Downloading global weather data ({start_year}-{end_year})")

        output_dir = self._ensure_weather_directory_exists(country)

        # Download each variable for each year
        for variable in variables:
            logger.info(f"Downloading {variable.key}...")
            for year in tqdm(range(start_year, end_year), desc=f"{variable.key}"):
                file_path = output_dir / f"{year}_{variable.key}.zip"
                
                if file_path.exists():
                    logger.debug(f"Skipping existing file: {file_path}")
                    continue
                
                request = self._build_cds_request(year, variable, bounds)
                logger.debug(f"CDS API Request: {request}")
                result = self.client.retrieve(self.config.dataset_name, request)
                
                # Download directly as zip file without extraction
                result.download(str(file_path))
                    
                logger.info(f"Downloaded: {file_path}")

    def _ensure_weather_directory_exists(self, country: Optional[str] = None) -> Path:
        """Create and return the weather data directory

        Args:
            country: Country name for subdirectory (optional)

        Returns:
            Path to the weather data directory
        """
        if country:
            # Create country-specific subdirectory
            weather_dir = self.data_dir / country.lower() / "weather"
        else:
            weather_dir = self.data_dir / "weather"
        weather_dir.mkdir(parents=True, exist_ok=True)
        return weather_dir

    def _build_cds_request(
        self,
        year: int,
        variable: WeatherVariable,
        bounds: Optional[GeoBounds] = None,
    ) -> dict:
        """Build a request dictionary for the AgERA5 CDS API

        Args:
            year: Year of data to request
            variable: Weather variable to download
            bounds: Geographic bounds to limit download area (optional)

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
        
        # Add area bounds for country-specific downloads
        if bounds:
            request["area"] = bounds.to_cds_area()
            
        return request
    

