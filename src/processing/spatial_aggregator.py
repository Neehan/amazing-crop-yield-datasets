"""Spatial aggregator for averaging gridded data over administrative boundaries"""

import logging
from typing import List
from pathlib import Path
import pickle

import pandas as pd
import xarray as xr
import numpy as np
from rasterio.transform import from_bounds
from tqdm import tqdm
from shapely.geometry import Point

from src.processing.base_processor import BaseProcessor


logger = logging.getLogger(__name__)


class SpatialAggregator:
    """Aggregates spatial data to administrative boundaries"""

    def __init__(self, base_processor: BaseProcessor, country_param: str):
        """Initialize with a base processor that provides boundaries"""
        self.base_processor = base_processor
        self.boundaries = base_processor.boundaries
        self.coverage_weights = None
        
        # Set up cache directory - use the same country string format as process_weather.py
        data_dir = Path("data") / country_param.lower() / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = data_dir

        logger.info(
            f"Spatial aggregator initialized with {len(self.boundaries)} boundaries"
        )
        logger.info(f"Cache directory: {self.cache_dir}")

    def aggregate_dataset(
        self, dataset: xr.Dataset
    ) -> pd.DataFrame:
        """Aggregate xarray dataset to administrative boundaries using weighted averaging

        Args:
            dataset: xarray Dataset with spatial data (daily weather data)

        Returns:
            DataFrame with time, admin_name, admin_id, value columns
        """
        logger.info(f"Aggregating spatial data using coverage-weighted averaging")

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

        # Load or compute coverage weights
        logger.info("Loading or computing coverage weights...")
        self.coverage_weights = self._get_or_compute_coverage_weights(transform, (len(lats), len(lons)), lats, lons)

        results = []

        # Process each boundary with its pre-computed coverage weights
        for i, (idx, boundary) in enumerate(tqdm(self.boundaries.iterrows(), total=len(self.boundaries), desc="Aggregating spatial data")):
            weights = self.coverage_weights[i]

            # Apply weighted aggregation - only over bounding box for efficiency
            aggregated_values = self._aggregate_weighted_data_optimized(
                var_data, weights, boundary.geometry, lat_coord, lon_coord
            )

            # Assign the variable name to the result
            aggregated_values.name = var_name

            # Convert to DataFrame - preserves time dimension
            admin_df = aggregated_values.to_dataframe().reset_index()

            # Add admin hierarchy info from GADM
            admin_df["country_name"] = boundary["COUNTRY"]
            admin_df["admin_level_1_name"] = boundary["NAME_1"]
            admin_df["admin_level_2_name"] = boundary["NAME_2"]
            admin_df["admin_id"] = idx
            
            # Add centroid coordinates (3 decimal places)
            centroid = boundary.geometry.centroid
            admin_df["latitude"] = round(centroid.y, 3)
            admin_df["longitude"] = round(centroid.x, 3)

            results.append(admin_df)

        # Combine all administrative units
        combined_df = pd.concat(results, ignore_index=True)
        logger.info(f"Aggregated data for {len(self.boundaries)} administrative units")
        logger.info(f"Result columns: {list(combined_df.columns)}")

        return combined_df

    def _get_or_compute_coverage_weights(self, transform, out_shape, lats, lons):
        """Load cached coverage weights or compute them with 25 subcells per grid cell"""
        
        cache_file = self.cache_dir / "coverage_weights_5x5_subcells.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached coverage weights from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Computing coverage weights with 25 subcells per grid cell...")
        coverage_weights = self._compute_coverage_weights(out_shape, lats, lons)
        
        # Cache the results
        logger.info(f"Caching coverage weights to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(coverage_weights, f)
        
        return coverage_weights

    def _compute_coverage_weights(self, out_shape, lats, lons):
        """Compute coverage weights by dividing each 0.1° grid cell into 25 subcells"""
        
        all_weights = []
        geometries = [boundary.geometry for _, boundary in self.boundaries.iterrows()]
        
        # Grid cell size (0.1 degrees)
        lat_step = abs(lats[1] - lats[0])
        lon_step = abs(lons[1] - lons[0])
        
        logger.info(f"Grid resolution: {lat_step:.3f}° x {lon_step:.3f}°")
        logger.info(f"Subcell resolution: {lat_step/5:.3f}° x {lon_step/5:.3f}°")
        
        for geom in tqdm(geometries, desc="Computing coverage weights"):
            weights = np.zeros(out_shape, dtype=np.float32)
            
            # Get bounding box
            bounds = geom.bounds  # minx, miny, maxx, maxy
            
            # Find grid cells that intersect the bounding box
            lat_indices = np.where(
                (lats >= bounds[1] - lat_step) & (lats <= bounds[3] + lat_step)
            )[0]
            lon_indices = np.where(
                (lons >= bounds[0] - lon_step) & (lons <= bounds[2] + lon_step)
            )[0]
            
            # For each grid cell in the bounding box, compute coverage
            for i in lat_indices:
                for j in lon_indices:
                    # Create 5x5 subcells within this grid cell
                    grid_lat = lats[i]
                    grid_lon = lons[j]
                    
                    # Grid cell bounds
                    cell_lat_min = grid_lat - lat_step/2
                    cell_lon_min = grid_lon - lon_step/2
                    
                    covered_subcells = 0
                    total_subcells = 25
                    
                    # Check each of the 25 subcells
                    for sub_i in range(5):
                        for sub_j in range(5):
                            # Subcell center
                            sub_lat = cell_lat_min + (sub_i + 0.5) * lat_step/5
                            sub_lon = cell_lon_min + (sub_j + 0.5) * lon_step/5
                            
                            # Create point for subcell center
                            subcell_point = Point(sub_lon, sub_lat)
                            
                            if geom.contains(subcell_point):
                                covered_subcells += 1
                    
                    # Coverage fraction for this grid cell
                    coverage = covered_subcells / total_subcells
                    weights[i, j] = coverage
            
            all_weights.append(weights)
        
        return all_weights
    
    def _aggregate_weighted_data_optimized(
        self,
        data: xr.DataArray,
        weights: np.ndarray,
        geometry,
        lat_coord: str = "lat",
        lon_coord: str = "lon",
    ) -> xr.DataArray:
        """Optimized weighted aggregation using coverage weights

        Args:
            data: Full xarray DataArray
            weights: Coverage weights for the full grid
            geometry: Shapely geometry for bounding box

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

        # Extract weights for this bounding box using the same indices
        clipped_weights = weights[
            lat_indices.min() : lat_indices.max() + 1,
            lon_indices.min() : lon_indices.max() + 1,
        ]

        # Convert weights to xarray with clipped coordinates
        weights_da = xr.DataArray(
            clipped_weights,
            dims=[lat_coord, lon_coord],
            coords={
                lat_coord: clipped_data[lat_coord],
                lon_coord: clipped_data[lon_coord],
            },
        )

        # Apply weights - only consider cells with non-zero weights
        valid_weights = weights_da > 0
        masked_data = clipped_data.where(valid_weights)
        masked_weights = weights_da.where(valid_weights)

        # Weighted average: sum(data * weights) / sum(weights)
        weighted_sum = (masked_data * masked_weights).sum(dim=[lat_coord, lon_coord], skipna=True)
        total_weights = masked_weights.sum(dim=[lat_coord, lon_coord], skipna=True)
        
        # Avoid division by zero
        result = weighted_sum / total_weights.where(total_weights > 0)

        return result

    def aggregate_file(self, file_path: str) -> pd.DataFrame:
        """Aggregate a single NetCDF file to administrative boundaries"""
        logger.info(f"Processing file: {file_path}")

        dataset = xr.open_dataset(file_path)
        result = self.aggregate_dataset(dataset)

        return result

    def aggregate_multiple_files(
        self, file_paths: List[str]
    ) -> pd.DataFrame:
        """Aggregate multiple NetCDF files to administrative boundaries"""
        logger.info(f"Processing {len(file_paths)} files")

        all_results = []

        for file_path in file_paths:
            result = self.aggregate_file(file_path)
            all_results.append(result)

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined results: {combined_df.shape}")

        return combined_df
