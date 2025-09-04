"""Soil data downloader using SoilGrids Python package"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from pyproj import Transformer
from soilgrids import SoilGrids
from src.downloader.base import BaseDownloader
from src.downloader.soil.models import (
    SoilProperty,
    SoilDepth,
    PROPERTY_DEPTH_MAPPING,
    MEAN_STATISTIC,
)

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

    def _convert_bounds_to_projected(self, bounds):
        """Convert lat/lon bounds to SoilGrids projected coordinates (Homolosine)

        SoilGrids uses Interrupted Goode Homolosine projection
        """
        # SoilGrids Homolosine CRS definition
        homolosine_crs = 'PROJCS["Interrupted_Goode_Homolosine",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Interrupted_Goode_Homolosine"],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

        # Create transformer from WGS84 to Homolosine
        transformer = Transformer.from_crs("EPSG:4326", homolosine_crs, always_xy=True)

        # Transform corner points
        west, south = transformer.transform(bounds.min_lon, bounds.min_lat)
        east, north = transformer.transform(bounds.max_lon, bounds.max_lat)

        return west, south, east, north

    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build SoilGrids download parameters"""
        soil_property = kwargs["property"]
        depth = kwargs["depth"]
        bounds = kwargs.get("bounds")
        file_path = kwargs["file_path"]

        # Build the service and coverage IDs
        service_id = soil_property.code
        coverage_id = f"{soil_property.code}_{depth.range_str}_{MEAN_STATISTIC}"

        # Convert bounds to projected coordinates
        west, south, east, north = self._convert_bounds_to_projected(bounds)

        return {
            "service_id": service_id,
            "coverage_id": coverage_id,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "crs": "urn:ogc:def:crs:EPSG::152160",  # SoilGrids Homolosine projection
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
