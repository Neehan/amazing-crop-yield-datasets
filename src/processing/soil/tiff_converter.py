"""Soil TIF to NetCDF converter - converts soil TIF files to combined NetCDF with depth dimension"""

import logging
from pathlib import Path
from typing import List, Optional
import re

import xarray as xr
import rasterio
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SoilTiffConverter:
    """Converter for soil TIF files to NetCDF with depth dimension"""

    def __init__(self, cache_dir: Path):
        """Initialize soil TIF converter

        Args:
            cache_dir: Directory to cache converted NetCDF files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def convert_property_to_netcdf(
        self, soil_dir: Path, property_name: str, depths: List[str]
    ) -> Path:
        """Convert TIF files for a property into NetCDF with depth dimension

        Args:
            soil_dir: Directory containing soil TIF files
            property_name: Soil property name (e.g., 'bulk_density')
            depths: List of depth ranges to include (e.g., ['0_5cm', '5_15cm'])

        Returns:
            Path to combined NetCDF file
        """
        cache_filename = f"{property_name}_combined_depths.nc"
        cache_path = self.cache_dir / cache_filename

        # Return cached file if it exists
        if cache_path.exists():
            logger.debug(f"Using cached NetCDF file: {cache_path}")
            return cache_path

        logger.info(f"Converting TIF files for property: {property_name}")

        pattern = f"{property_name}_*_mean.tif"
        tif_files = list(soil_dir.glob(pattern))

        if not tif_files:
            raise ValueError(
                f"No TIF files found for property {property_name} in {soil_dir}"
            )

        logger.info(f"Found {len(tif_files)} TIF files for {property_name}")

        # Extract depth info and sort
        depth_files = []
        for tif_file in tif_files:
            depth_key = self._extract_depth_from_filename(tif_file.name, property_name)
            if depth_key and depth_key in depths:
                depth_files.append((depth_key, tif_file))

        if not depth_files:
            raise ValueError(
                f"No matching depth files found for {property_name} with depths {depths}"
            )

        depth_files.sort(key=lambda x: self._get_depth_sort_key(x[0]))

        logger.info(f"Processing depths: {[depth for depth, _ in depth_files]}")

        # Load each TIF and combine
        data_arrays = []
        depth_coords = []

        for depth_key, tif_file in tqdm(
            depth_files, desc=f"Loading {property_name} TIFs"
        ):
            with rasterio.open(tif_file) as src:
                data = src.read(1)
                transform = src.transform
                crs = src.crs
                nodata = src.nodata

                height, width = data.shape
                x_coords = np.array(
                    [transform[2] + transform[0] * (i + 0.5) for i in range(width)]
                )
                y_coords = np.array(
                    [transform[5] + transform[4] * (j + 0.5) for j in range(height)]
                )

                da = xr.DataArray(
                    data,
                    coords={"lat": (["lat"], y_coords), "lon": (["lon"], x_coords)},
                    dims=["lat", "lon"],
                    attrs={"crs": str(crs), "nodata": nodata, "depth_range": depth_key},
                )

                if nodata is not None:
                    da = da.where(da != nodata)

                data_arrays.append(da)
                depth_coords.append(depth_key)

        # Combine along depth dimension
        combined_da = xr.concat(data_arrays, dim="depth")
        combined_da = combined_da.assign_coords(depth=depth_coords)

        dataset = xr.Dataset({property_name: combined_da})

        dataset.attrs.update(
            {
                "title": f"Soil {property_name} data",
                "source": "SoilGrids",
                "property": property_name,
                "depths": depth_coords,
                "crs": str(crs) if "crs" in locals() else "EPSG:4326",
            }
        )

        # Save to cache
        logger.info(f"Saving NetCDF file: {cache_path}")
        dataset.to_netcdf(cache_path)

        logger.info(f"Combined dataset shape: {dataset[property_name].shape}")
        return cache_path

    def _extract_depth_from_filename(
        self, filename: str, property_name: str
    ) -> Optional[str]:
        """Extract depth key from filename"""
        remaining = filename.replace(f"{property_name}_", "").replace("_mean.tif", "")

        if re.match(r"\d+_\d+cm$", remaining):
            return remaining

        logger.warning(f"Could not extract depth from filename: {filename}")
        return None

    def _get_depth_sort_key(self, depth_key: str) -> int:
        """Get numeric sort key for depth range"""
        start_depth = int(depth_key.split("_")[0])
        return start_depth
