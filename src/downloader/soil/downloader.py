"""Soil data downloader using SoilGrids Python package"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from soilgrids import SoilGrids
from src.downloader.base import BaseDownloader
from src.downloader.soil.models import (
    SoilProperty,
    SoilDepth,
    PROPERTY_DEPTH_MAPPING,
    MEAN_STATISTIC,
)
from src.constants import SOIL_RESOLUTION_DEGREES

# Set up logger
logger = logging.getLogger(__name__)


class SoilDownloader(BaseDownloader):
    """Downloads soil data from SoilGrids using the soilgrids Python package"""

    def __init__(self, data_dir: str, country: str, max_concurrent: int):
        """Initialize soil downloader with SoilGrids package

        Args:
            data_dir: Base directory to save downloaded files
            country: Country name for subdirectory
            max_concurrent: Maximum concurrent downloads
        """
        super().__init__(data_dir, country, "soil", max_concurrent)
        self.soil_grids = SoilGrids()

    def _get_geographic_bounds(self, bounds):
        """Return bounds in geographic coordinates (WGS84)

        No projection needed - keep in lat/lon for reasonable resolution
        """
        return bounds.min_lon, bounds.min_lat, bounds.max_lon, bounds.max_lat

    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build SoilGrids download parameters"""
        soil_property = kwargs["property"]
        depth = kwargs["depth"]
        bounds = kwargs.get("bounds")
        file_path = kwargs["file_path"]

        # Build the service and coverage IDs
        service_id = soil_property.code
        coverage_id = f"{soil_property.code}_{depth.range_str}_{MEAN_STATISTIC}"

        # Use geographic coordinates (WGS84) with configurable resolution
        west, south, east, north = self._get_geographic_bounds(bounds)

        # Calculate width and height based on resolution from constants
        lon_range = east - west
        lat_range = north - south
        width = int(lon_range / SOIL_RESOLUTION_DEGREES)
        height = int(lat_range / SOIL_RESOLUTION_DEGREES)

        return {
            "service_id": service_id,
            "coverage_id": coverage_id,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "width": width,
            "height": height,
            "crs": "urn:ogc:def:crs:EPSG::4326",  # WGS84 geographic coordinates
            "output": str(file_path),
        }

    def _make_api_request(self, **kwargs) -> Any:
        """Make SoilGrids request using the soilgrids package"""
        return self.soil_grids.get_coverage_data(**kwargs)

    async def _handle_api_response(self, result: Any, **kwargs):
        """Handle SoilGrids response - file is already saved by the package"""
        file_path = kwargs.get("file_path")
        logger.debug(f"Downloaded soil data to: {file_path}")

    async def download_soil(
        self,
        properties: Optional[List[SoilProperty]] = None,
        depths: Optional[List[SoilDepth]] = None,
    ):
        """Download soil data from SoilGrids (mean values only)

        Args:
            properties: Soil properties to download (default: all available)
            depths: Depth ranges to download (default: all available for each property)
        """
        properties = properties or list(SoilProperty)

        # Get country bounds for spatial filtering
        bounds = self.get_country_bounds()

        await self._download_individual_files(properties, depths, bounds)

    async def _download_individual_files(self, properties, depths, bounds):
        """Download individual soil property/depth files"""
        download_tasks = []

        for soil_property in properties:
            # Get available depths for this property
            available_depths = PROPERTY_DEPTH_MAPPING.get(soil_property, [])
            property_depths = depths or available_depths

            # Only use depths that are actually available for this property
            valid_depths = [d for d in property_depths if d in available_depths]

            if not valid_depths:
                logger.warning(
                    f"No valid depths found for property {soil_property.code}"
                )
                continue

            for depth in valid_depths:
                filename = f"{soil_property.key}_{depth.key}_mean.tif"
                file_path = self.subdir / filename

                # Skip if file already exists
                if file_path.exists():
                    logger.info(f"File already exists: {file_path}")
                    continue

                download_tasks.append(
                    {
                        "property": soil_property,
                        "depth": depth,
                        "bounds": bounds,
                        "file_path": file_path,
                        "item_id": f"soil_{soil_property.code}_{depth.key}_mean",
                    }
                )

        if download_tasks:
            logger.info(f"Downloading {len(download_tasks)} soil files")
            await self.download(download_tasks, "soil data")
