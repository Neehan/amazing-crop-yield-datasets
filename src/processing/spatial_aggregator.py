"""Spatial aggregator for averaging gridded data over administrative boundaries"""

import logging
from typing import List

import pandas as pd
import xarray as xr
import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

from src.processing.base_processor import BaseProcessor


logger = logging.getLogger(__name__)


class SpatialAggregator:
    """Aggregates spatial data to administrative boundaries"""

    def __init__(self, base_processor: BaseProcessor):
        """Initialize with a base processor that provides boundaries"""
        self.base_processor = base_processor
        self.boundaries = base_processor.boundaries

        logger.info(
            f"Spatial aggregator initialized with {len(self.boundaries)} boundaries"
        )

    def aggregate_dataset(
        self, dataset: xr.Dataset, method: str = "mean"
    ) -> pd.DataFrame:
        """Aggregate xarray dataset to administrative boundaries using optimized zonal stats

        Args:
            dataset: xarray Dataset with spatial data (daily weather data)
            method: Aggregation method (mean, sum, median)

        Returns:
            DataFrame with time, admin_name, admin_id, value columns
        """
        logger.info(f"Aggregating spatial data using {method} method")

        # Get the main data variable (filter out metadata like crs)
        data_vars = [
            var for var in dataset.data_vars if var not in ["crs", "spatial_ref"]
        ]
        assert (
            len(data_vars) == 1
        ), f"Expected 1 data variable after filtering, got {len(data_vars)}: {data_vars}"

        var_name = data_vars[0]
        var_data = dataset[var_name]

        # Get coordinate arrays once - handle different coordinate names
        coord_names = list(var_data.coords)
        logger.debug(f"Available coordinates: {coord_names}")

        # Find latitude coordinate (could be 'latitude', 'lat', 'y')
        lat_coord = None
        for coord in ["latitude", "lat", "y"]:
            if coord in coord_names:
                lat_coord = coord
                break
        assert lat_coord, f"No latitude coordinate found in {coord_names}"

        # Find longitude coordinate (could be 'longitude', 'lon', 'x')
        lon_coord = None
        for coord in ["longitude", "lon", "x"]:
            if coord in coord_names:
                lon_coord = coord
                break
        assert lon_coord, f"No longitude coordinate found in {coord_names}"

        lats = var_data[lat_coord].values
        lons = var_data[lon_coord].values

        # Create transform for the full grid once
        transform = from_bounds(
            lons.min(), lats.min(), lons.max(), lats.max(), len(lons), len(lats)
        )

        # Create all masks at once - more efficient
        logger.info("Creating polygon masks for all boundaries...")
        all_masks = self._create_all_masks(transform, (len(lats), len(lons)))

        results = []

        # Process each boundary with its pre-computed mask
        for i, (idx, boundary) in enumerate(self.boundaries.iterrows()):
            logger.debug(f"Processing boundary {i+1}/{len(self.boundaries)}")

            mask = all_masks[i]

            # Apply mask and aggregate - only over bounding box for efficiency
            aggregated_values = self._aggregate_masked_data_optimized(
                var_data, mask, boundary.geometry, method, lat_coord, lon_coord
            )

            # Convert to DataFrame - preserves time dimension
            admin_df = aggregated_values.to_dataframe().reset_index()

            # Add admin hierarchy info from GADM
            admin_df["country_name"] = boundary["COUNTRY"]
            admin_df["admin_level_1_name"] = boundary["NAME_1"]
            admin_df["admin_level_2_name"] = boundary["NAME_2"]
            admin_df["admin_id"] = idx

            results.append(admin_df)

        # Combine all administrative units
        combined_df = pd.concat(results, ignore_index=True)
        logger.info(f"Aggregated data for {len(self.boundaries)} administrative units")
        logger.info(f"Result columns: {list(combined_df.columns)}")

        return combined_df

    def _create_all_masks(self, transform, out_shape):
        """Create masks for all polygons at once - more efficient than one by one"""
        logger.info(f"Creating masks for {len(self.boundaries)} polygons...")

        all_masks = []
        geometries = [boundary.geometry for _, boundary in self.boundaries.iterrows()]

        # Create all masks in batch
        for geom in geometries:
            mask = geometry_mask(
                [geom], out_shape=out_shape, transform=transform, invert=False
            )
            all_masks.append(mask)

        return all_masks

    def _aggregate_masked_data_optimized(
        self,
        data: xr.DataArray,
        mask: np.ndarray,
        geometry,
        method: str,
        lat_coord: str = "lat",
        lon_coord: str = "lon",
    ) -> xr.DataArray:
        """Optimized masked aggregation - clip to bounding box first, then apply mask

        Args:
            data: Full xarray DataArray
            mask: Boolean mask for the full grid
            geometry: Shapely geometry for bounding box
            method: Aggregation method

        Returns:
            Aggregated data with time dimension preserved
        """
        # Get bounding box for this geometry
        bounds = geometry.bounds  # minx, miny, maxx, maxy

        # Clip data to bounding box for efficiency
        # Find indices for bounding box
        lat_indices = np.where(
            (data[lat_coord] >= bounds[1]) & (data[lat_coord] <= bounds[3])
        )[0]
        lon_indices = np.where(
            (data[lon_coord] >= bounds[0]) & (data[lon_coord] <= bounds[2])
        )[0]

        if len(lat_indices) == 0 or len(lon_indices) == 0:
            # Return NaN array with same time dimension
            return data.isel({lat_coord: slice(0, 0), lon_coord: slice(0, 0)}).mean(
                dim=[lat_coord, lon_coord]
            )

        clipped_data = data.isel(
            {
                lat_coord: slice(lat_indices.min(), lat_indices.max() + 1),
                lon_coord: slice(lon_indices.min(), lon_indices.max() + 1),
            }
        )

        # Extract mask for this bounding box using the same indices
        clipped_mask = mask[
            lat_indices.min() : lat_indices.max() + 1,
            lon_indices.min() : lon_indices.max() + 1,
        ]

        # Convert mask to xarray with clipped coordinates
        mask_da = xr.DataArray(
            clipped_mask,
            dims=[lat_coord, lon_coord],
            coords={
                lat_coord: clipped_data[lat_coord],
                lon_coord: clipped_data[lon_coord],
            },
        )

        # Apply mask - set values outside polygon to NaN
        masked_data = clipped_data.where(~mask_da)

        # Aggregate over spatial dimensions, preserving time
        if method == "mean":
            result = masked_data.mean(dim=[lat_coord, lon_coord], skipna=True)
        elif method == "sum":
            result = masked_data.sum(dim=[lat_coord, lon_coord], skipna=True)
        elif method == "median":
            result = masked_data.median(dim=[lat_coord, lon_coord], skipna=True)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        return result

    def aggregate_file(self, file_path: str, method: str = "mean") -> pd.DataFrame:
        """Aggregate a single NetCDF file to administrative boundaries"""
        logger.info(f"Processing file: {file_path}")

        dataset = xr.open_dataset(file_path)
        result = self.aggregate_dataset(dataset, method)

        return result

    def aggregate_multiple_files(
        self, file_paths: List[str], method: str = "mean"
    ) -> pd.DataFrame:
        """Aggregate multiple NetCDF files to administrative boundaries"""
        logger.info(f"Processing {len(file_paths)} files")

        all_results = []

        for file_path in file_paths:
            result = self.aggregate_file(file_path, method)
            all_results.append(result)

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined results: {combined_df.shape}")

        return combined_df
