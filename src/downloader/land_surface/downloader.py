"""Land surface data downloader"""

import asyncio
import cdsapi
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.downloader.base import BaseDownloader
from src.downloader.land_surface.models import DownloadConfig, LandSurfaceVariable

# Set up logger
logger = logging.getLogger(__name__)


class LandSurfaceDownloader(BaseDownloader):
    """Downloads land surface data from ERA5 single levels daily statistics via CDS API"""

    def __init__(self, data_dir: str, country: str, max_concurrent: int):
        """Initialize land surface downloader with CDS API client

        Args:
            data_dir: Base directory to save downloaded files
            country: Country name for subdirectory
            max_concurrent: Maximum concurrent downloads
        """
        super().__init__(data_dir, country, "land_surface", max_concurrent)
        self.config = DownloadConfig()
        self.client = cdsapi.Client()

    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build CDS API request parameters"""
        year = kwargs["year"]
        variable = kwargs["variable"]
        bounds = kwargs.get("bounds")

        request = {
            "variable": [variable.variable],
            "year": str(year),
            "month": [f"{month:02d}" for month in range(1, 13)],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "daily_statistic": self.config.daily_statistic,
            "time_zone": self.config.time_zone,
            "frequency": self.config.frequency,
        }

        if self.config.product_type is not None:
            request["product_type"] = self.config.product_type

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

    async def download_land_surface(
        self,
        start_year: int,
        end_year: Optional[int],
        variables: Optional[List[LandSurfaceVariable]],
    ):
        """Download ERA5 land surface data with concurrent processing

        Args:
            start_year: First year to download (inclusive)
            end_year: Last year to download (exclusive, default: current year)
            variables: Land surface variables to download (default: all available)
        """
        end_year = end_year or datetime.now().year
        variables = variables or list(LandSurfaceVariable)

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
        await self.download(download_tasks, f"Land surface data for {self.country}")
