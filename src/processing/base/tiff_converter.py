"""Simple TIF to xarray converter using rioxarray"""

import logging
from pathlib import Path
from typing import List, Optional
import datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import ColorInterp
import xarray as xr
import rioxarray as rxr
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress rasterio warnings about TIFF metadata issues
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)


class TiffConverter:
    """Simple converter for GeoTIFF files to xarray format"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def convert_tiff_to_weekly_nc(
        self, tiff_path: Path, variable: str, year: int
    ) -> Path:
        """Convert GeoTIFF to weekly NetCDF - dead simple with rioxarray"""
        cache_path = self.cache_dir / f"{year}_{variable}_weekly.nc"

        if cache_path.exists():
            return cache_path

            # Read TIF and fix metadata corruption issues by filtering data bands
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_LOG_LEVEL="ERROR"
        ):
            with rasterio.open(tiff_path) as src:
                # Keep only meaningful data bands (gray or undefined); drop alpha/extra
                keep = [
                    i + 1
                    for i, ci in enumerate(src.colorinterp)
                    if ci
                    in (
                        ColorInterp.gray,
                        ColorInterp.undefined,
                        ColorInterp.red,
                        ColorInterp.green,
                        ColorInterp.blue,
                    )
                ]
                if not keep:
                    # fallback: keep all bands (some files may have all bands as 'palette' or other types)
                    keep = list(range(1, src.count + 1))

                data = src.read(indexes=keep)
                transform = src.transform
                crs = src.crs
                height, width = src.height, src.width

        # Clean and validate data
        data = self._clean_and_validate_data(data, variable)

        # Build time coordinates (should always be 52 weeks now that downloader is fixed)
        n_bands = data.shape[0]
        jan1 = pd.Timestamp(year=year, month=1, day=1)

        if n_bands != 52:
            logger.warning(
                f"Year {year} {variable}: Expected 52 weeks, got {n_bands} - this should not happen with fixed downloader!"
            )

        times = pd.date_range(jan1, periods=52, freq="7D")

        # Create clean DataArray with proper spatial coordinates from transform
        lons, lats = np.meshgrid(
            np.linspace(transform.c, transform.c + transform.a * width, width),
            np.linspace(transform.f, transform.f + transform.e * height, height),
        )

        da = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={
                "time": times,
                "lat": lats[:, 0],  # Take first column for lat coords
                "lon": lons[0, :],  # Take first row for lon coords
            },
        )
        da = da.rio.write_crs(crs)

        ds = xr.Dataset({variable: da})
        ds.to_netcdf(cache_path)

        return cache_path

    def process_all_tiffs(
        self,
        land_surface_dir: Path,
        year_range: tuple,
        variables: Optional[List[str]] = None,
    ) -> List[Path]:
        """Process TIF files in directory"""
        start_year, end_year = year_range
        tiff_files = list(land_surface_dir.glob("*.tif"))

        weekly_files = []
        for tiff_path in tqdm(tiff_files, desc="Converting TIFs"):
            # Parse filename: "2020_lai_low_weekly.tif" -> year=2020, variable="lai_low"
            parts = tiff_path.stem.split("_")
            year = int([p for p in parts if p.isdigit() and len(p) == 4][0])

            if not (start_year <= year < end_year):
                continue

            # Get variable name (everything between year and "weekly")
            year_idx = parts.index(str(year))
            weekly_idx = parts.index("weekly") if "weekly" in parts else len(parts)
            variable = "_".join(parts[year_idx + 1 : weekly_idx])

            if variables and variable not in variables:
                continue

            weekly_nc_path = self.convert_tiff_to_weekly_nc(tiff_path, variable, year)
            weekly_files.append(weekly_nc_path)

        return weekly_files

    def _clean_and_validate_data(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Clean and validate data based on variable type

        Args:
            data: Raw data array from TIFF
            variable: Variable name (e.g., 'ndvi', 'lai_high')

        Returns:
            Cleaned data array with invalid values set to NaN
        """
        # Convert -inf values (over water) and -999999 (missing weeks) to NaN
        data = np.where(np.isneginf(data), np.nan, data)
        data = np.where(data == -999999, np.nan, data)

        # Apply variable-specific data validation
        variable_lower = variable.lower()

        if variable_lower == "ndvi":
            # NOAA NDVI is typically scaled by 10000, so valid range is -10000 to 10000
            # Filter extreme outliers that are clearly invalid
            data = np.where((data >= -10000) & (data <= 10000), data, np.nan)
        elif variable_lower.startswith("lai"):
            # LAI should be between 0 and 8, filter extreme outliers
            data = np.where((data >= 0) & (data <= 9), data, np.nan)
        # Add more variable validations here as needed
        return data
