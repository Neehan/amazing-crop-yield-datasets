"""Weather data downloader"""

import cdsapi
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from tqdm import tqdm
import asyncio

from src.weather.models import WeatherVariable, DownloadConfig
from src.utils.geography import Geography, GeoBounds

# Set up logger
logger = logging.getLogger(__name__)


class WeatherDownloader:
    """Downloads weather data from AgERA5 climate dataset via CDS API"""

    def __init__(self, data_dir: str = "data", max_concurrent: int = 5):
        """Initialize downloader with data directory and API client

        Args:
            data_dir: Base directory to save downloaded files
            max_concurrent: Maximum concurrent downloads (default: 5)
        """
        self.data_dir = Path(data_dir)
        self.config = DownloadConfig()
        self.geography = Geography()
        self.max_concurrent = max_concurrent
        self.client = cdsapi.Client()

    async def download(
        self,
        start_year: int = 1979,
        end_year: Optional[int] = None,
        variables: Optional[List[WeatherVariable]] = None,
        country: Optional[str] = None,
    ):
        """Download weather data (global or country-specific)

        Args:
            start_year: First year to download data for (inclusive)
            end_year: Last year to download data for (exclusive, defaults to current year)
            variables: List of weather variables to download (defaults to all)
            country: Country name to filter data (defaults to global)
        """
        end_year = end_year or datetime.now().year
        variables = variables or list(WeatherVariable)

        bounds = self._get_country_bounds(country)
        output_dir = self._ensure_weather_directory_exists(country)

        for variable in variables:
            await self._download_variable(
                variable, start_year, end_year, bounds, output_dir
            )

    def _get_country_bounds(self, country: Optional[str]) -> Optional[GeoBounds]:
        """Get country bounds and log download info"""
        if country:
            bounds = self.geography.get_country_bounds(country)
            logger.info(f"Downloading weather data for {country}")
            return bounds
        else:
            logger.info("Downloading global weather data")
            return None

    async def _download_variable(
        self,
        variable: WeatherVariable,
        start_year: int,
        end_year: int,
        bounds: Optional[GeoBounds],
        output_dir: Path,
    ):
        """Download all years for a single variable concurrently"""
        logger.info(f"Downloading {variable.key}...")

        years_to_download = self._get_missing_years(
            variable, start_year, end_year, output_dir
        )

        if not years_to_download:
            logger.info(f"All files for {variable.key} already exist, skipping")
            return

        logger.info(
            f"Downloading {len(years_to_download)} years for {variable.key} with {self.max_concurrent} concurrent downloads"
        )

        await self._download_years_concurrent(
            variable, years_to_download, bounds, output_dir
        )

    def _get_missing_years(
        self,
        variable: WeatherVariable,
        start_year: int,
        end_year: int,
        output_dir: Path,
    ) -> List[int]:
        """Get list of years that need to be downloaded"""
        years_to_download = []
        for year in range(start_year, end_year):
            file_path = output_dir / f"{year}_{variable.key}.zip"
            if file_path.exists():
                logger.debug(f"Skipping existing file: {file_path}")
            else:
                years_to_download.append(year)
        return years_to_download

    async def _download_years_concurrent(
        self,
        variable: WeatherVariable,
        years: List[int],
        bounds: Optional[GeoBounds],
        output_dir: Path,
    ):
        """Download multiple years concurrently with progress tracking"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        progress_bar = tqdm(total=len(years), desc=f"{variable.key}")

        async def download_year_with_progress(year):
            success = await self._download_single_year(
                semaphore, variable, year, bounds, output_dir
            )
            progress_bar.update(1)
            return success

        tasks = [download_year_with_progress(year) for year in years]
        await asyncio.gather(*tasks, return_exceptions=True)
        progress_bar.close()

    async def _download_single_year(
        self,
        semaphore: asyncio.Semaphore,
        variable: WeatherVariable,
        year: int,
        bounds: Optional[GeoBounds],
        output_dir: Path,
    ) -> bool:
        """Download data for a single year"""
        async with semaphore:
            try:
                file_path = output_dir / f"{year}_{variable.key}.zip"

                if file_path.exists():
                    return True

                request = self._build_cds_request(year, variable, bounds)
                logger.debug(f"CDS API Request for {year}: {request}")

                # Run blocking CDS calls in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.retrieve(self.config.dataset_name, request),
                )
                await loop.run_in_executor(None, result.download, str(file_path))

                logger.debug(f"Downloaded: {file_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to download {variable.key} for {year}: {e}")
                return False

    def _ensure_weather_directory_exists(self, country: Optional[str] = None) -> Path:
        """Create and return the weather data directory

        Args:
            country: Country name for subdirectory (optional)

        Returns:
            Path to the weather data directory
        """
        if country:
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
            "version": self.config.version,
        }

        # Add statistic for variables that require it
        if variable.statistic is not None:
            request["statistic"] = [variable.statistic]

        # Add area bounds for country-specific downloads
        if bounds:
            request["area"] = bounds.to_cds_area()

        return request
