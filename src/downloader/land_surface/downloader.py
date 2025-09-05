"""Land surface data downloader using Google Earth Engine"""

import asyncio
import ee

from ee.imagecollection import ImageCollection
from ee.image import Image
from ee.featurecollection import FeatureCollection
from ee.filter import Filter
from ee.ee_date import Date


import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.downloader.base import BaseDownloader
from src.downloader.land_surface.models import GEEConfig, LandSurfaceVariable

# Set up logger
logger = logging.getLogger(__name__)


class LandSurfaceDownloader(BaseDownloader):
    """Downloads land surface data from ERA5-Land via Google Earth Engine"""

    def __init__(self, data_dir: str, country: str, max_concurrent: int):
        """Initialize land surface downloader with Google Earth Engine"""
        super().__init__(data_dir, country, "land_surface", max_concurrent)
        self.config = GEEConfig()
        ee.Initialize()

    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build Google Earth Engine export parameters"""
        year = kwargs["year"]
        variable = kwargs["variable"]
        region = kwargs["region"]

        if year < 1982 and variable == "NDVI":
            raise ValueError("NDVI data is only available from 1982 onwards")

        start_date = f"{year}-01-01"
        end_date = f"{year+1}-01-01"

        # Select dataset based on variable and year
        if variable == "NDVI":
            dataset = self.config.get_ndvi_dataset(year)
        else:
            dataset = self.config.dataset

        return {
            "dataset": dataset,
            "start_date": start_date,
            "end_date": end_date,
            "variable": variable,
            "region": region,
            "year": year,
        }

    def _make_api_request(self, **kwargs) -> Any:
        """Create Google Earth Engine export task"""
        dataset = kwargs["dataset"]
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]
        variable = kwargs["variable"]
        region = kwargs["region"]
        year = kwargs["year"]

        ic = (
            ImageCollection(dataset)
            .filterDate(start_date, end_date)
            .select(variable)
            .map(lambda img: img.clip(region))
        )

        # Create weekly composites (52 bands) for the specified variable
        variable = kwargs["variable"]  # Single variable passed from task

        weekly_images = []
        for week in range(52):
            # Calculate start and end dates for this week
            week_start = Date(start_date).advance(week * 7, "day")
            week_end = week_start.advance(7, "day")

            # Filter collection to this week
            week_collection = ic.filterDate(week_start, week_end).select(variable)

            # Check if we have any data for this week
            week_size = week_collection.size()

            # Create weekly mean, handling missing data properly
            weekly_mean = ee.Algorithms.If(
                week_size.gt(0),
                # If we have data, compute mean
                week_collection.mean().rename(f"week_{week+1:02d}"),
                # If no data, create image with -999999 (large negative number for missing data)
                Image.constant(-999999).rename(f"week_{week+1:02d}").clip(region),
            )

            weekly_images.append(Image(weekly_mean))

        # Combine all weekly means into one multi-band image
        multi_band_image = Image.cat(weekly_images).toFloat()

        return multi_band_image

    async def _handle_api_response(self, image: Any, **kwargs):
        """Handle Google Earth Engine image by downloading directly"""
        file_path = kwargs["file_path"]
        year = kwargs["year"]
        variable_name = kwargs["variable_name"]

        logger.info(f"Downloading {variable_name} for {year}")

        # Get download URL from the image
        loop = asyncio.get_event_loop()
        download_url = await loop.run_in_executor(
            None,
            lambda: image.getDownloadURL(
                {
                    "region": kwargs["region"],
                    "scale": self.config.scale_meters,
                    "crs": self.config.crs,
                    "format": "GEO_TIFF",
                }
            ),
        )

        # Download file to local path
        import requests

        response = requests.get(download_url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded image to: {file_path}")

    def _get_country_region(self) -> Any:
        """Get country geometry from Google Earth Engine"""
        # Use base class geography instance to get proper country name
        proper_country_name = self.geography.get_country_full_name(self.country)

        gaul0 = FeatureCollection(self.config.boundaries_dataset)
        country_feature = gaul0.filter(
            Filter.eq(self.config.country_property, proper_country_name)
        ).first()
        return country_feature.geometry()

    async def download_land_surface(
        self,
        start_year: int,
        end_year: Optional[int],
        variables: Optional[List[LandSurfaceVariable]],
    ):
        """Download ERA5-Land data via Google Earth Engine"""
        end_year = end_year or datetime.now().year
        variables = variables or list(LandSurfaceVariable)

        # Log which variables are being downloaded
        variable_names = [var.key_suffix for var in variables]
        logger.info(
            f"Downloading variables: {', '.join(variable_names)} for {self.country} ({start_year}-{end_year-1})"
        )

        region = self._get_country_region()

        download_tasks = []
        for year in range(start_year, end_year):
            for variable in variables:
                file_path = self.subdir / f"{year}_{variable.key_suffix}_weekly.tif"

                if not file_path.exists():
                    download_tasks.append(
                        {
                            "year": year,
                            "variable": variable.variable,  # Pass the band name
                            "variable_name": variable.key_suffix,  # Pass readable name for logging
                            "region": region,
                            "file_path": file_path,
                            "item_id": f"land_surface_{year}_{variable.key_suffix}",
                        }
                    )

        if not download_tasks:
            logger.info("All files already exist, skipping download")
            return

        logger.info(
            f"Starting {len(download_tasks)} GEE export tasks for {self.country}"
        )

        await self.download(download_tasks, f"Land surface data for {self.country}")
