"""
CCI Cropland Mask Downloader

Downloads satellite land cover data from Copernicus Climate Data Store (CDS).
Data is saved per country in data/country/raw/cci_cropland_mask/ directory.
"""

import asyncio
import cdsapi
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.downloader.base import BaseDownloader
from src.downloader.cci_cropland_mask.models import (
    CCICroplandMaskConfig,
    CCICroplandMaskVariable,
)

# Set up logger
logger = logging.getLogger(__name__)


class CCICroplandMaskDownloader(BaseDownloader):
    """Downloads CCI Cropland Mask data from CDS."""

    def __init__(self, data_dir: str, country: str, max_concurrent: int):
        """Initialize cci cropland mask downloader with CDS API client

        Args:
            data_dir: Base directory to save downloaded files
            country: Country name for subdirectory
            max_concurrent: Maximum concurrent downloads
        """
        super().__init__(data_dir, country, "cci_cropland_mask", max_concurrent)
        self.config = CCICroplandMaskConfig()
        self.client = cdsapi.Client()

    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build CDS API request parameters"""
        year = kwargs["year"]
        variable = kwargs["variable"]
        bounds = kwargs.get("bounds")

        request = {
            "variable": variable.variable,
            "year": [str(year)],
            "version": self.config.versions,
        }

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

    async def download_cci_cropland_mask(
        self,
        start_year: int,
        end_year: Optional[int],
        variables: Optional[List[CCICroplandMaskVariable]],
    ):
        """Download CCI Cropland Mask data with concurrent processing

        Args:
            start_year: First year to download (inclusive)
            end_year: Last year to download (exclusive, default: current year)
            variables: CCI cropland mask variables to download (default: all available)
        """
        end_year = end_year or datetime.now().year
        variables = variables or list(CCICroplandMaskVariable)

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
        await self.download(
            download_tasks, f"CCI cropland mask data for {self.country}"
        )
