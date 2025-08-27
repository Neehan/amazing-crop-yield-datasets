"""Spatial aggregator for averaging gridded data over administrative boundaries"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path
import pickle
import cftime

import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
import rasterio.features
from rasterio.transform import from_bounds

from src.processing.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class SpatialAggregator:
    """Aggregates spatial data to administrative boundaries with cropland filtering

    Two-stage process:
    1. FILTERING: Create masks to identify which grid cells to process
       - Admin boundary mask: Which admin unit each grid cell belongs to (cached globally)
       - Cropland mask: Which grid cells have cropland (cached per year)
       - Combined: Only process admin units that have cropland

    2. AVERAGING: For each admin unit, compute area-weighted average
       - Uses 25-subcell subdivision per grid cell for proper area weighting
       - Combines area weights with cropland filtering for final aggregation
    """

    def __init__(self, base_processor: BaseProcessor, country_param: str):
        """Initialize with a base processor that provides boundaries"""
        self.base_processor = base_processor
        self.boundaries = base_processor.boundaries
        self.country_param = country_param

        # Set up cache directory - use the same country string format as process_weather.py
        cropland_mask_dir = Path("data") / country_param.lower() / "cropland_mask"
        cropland_mask_dir.mkdir(parents=True, exist_ok=True)
        self.cropland_mask_dir = cropland_mask_dir

        # Load HYDE cropland data once
        self._load_cropland_data()

        # Cache for admin boundary mask (independent of year)
        self.admin_mask_cache = None

        # Cache for coverage weights (area-weighted averaging)
        self.coverage_weights = None

        logger.info(
            f"Spatial aggregator initialized with {len(self.boundaries)} boundaries"
        )
        logger.info(
            "Cropland filtering enabled (admin units with no cropland will be skipped)"
        )

    def aggregate_dataset(self, dataset: xr.Dataset) -> pd.DataFrame:
        """Aggregate xarray dataset to administrative boundaries using weighted averaging

        Args:
            dataset: xarray Dataset with spatial data (daily weather data)

        Returns:
            DataFrame with time, admin_name, admin_id, value columns
        """
        # Extract all years from time dimension for per-year cropland filtering
        all_years = self._extract_all_years_from_dataset(dataset)
        logger.info(
            f"Dataset spans years {min(all_years)} to {max(all_years)} - will use per-year cropland filtering"
        )

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

        # Calculate grid step sizes for transform bounds
        lat_step = abs(lats[1] - lats[0])
        lon_step = abs(lons[1] - lons[0])

        # STEP 1: FILTERING - Pre-compute admin mask (once) and all year cropland masks
        # Admin mask: computed once and cached globally since boundaries don't change
        admin_mask = self._get_or_compute_admin_mask(lats, lons)

        # Cropland masks: pre-compute for all years and cache together
        all_cropland_masks = self._get_or_compute_all_cropland_masks(
            lats, lons, all_years
        )

        # STEP 2: AVERAGING - Get coverage weights for area-weighted averaging
        logger.info("Loading or computing coverage weights...")
        transform = from_bounds(
            lons.min() - lon_step / 2,
            lats.min() - lat_step / 2,
            lons.max() + lon_step / 2,
            lats.max() + lat_step / 2,
            len(lons),
            len(lats),
        )
        self.coverage_weights = self._get_or_compute_coverage_weights(
            transform, (len(lats), len(lons)), lats, lons
        )

        # STEP 3: VECTORIZED AGGREGATION - Process all admin units simultaneously
        logger.info("Starting vectorized spatial aggregation...")
        result_df = self._aggregate_all_boundaries_vectorized(
            var_data,
            admin_mask,
            all_cropland_masks,
            str(var_name),
            lat_coord,
            lon_coord,
        )

        logger.info(
            f"Vectorized aggregation completed for {len(result_df['admin_id'].unique())} admin units"
        )
        logger.info(f"Result columns: {list(result_df.columns)}")

        return result_df

    def _aggregate_all_boundaries_vectorized(
        self,
        var_data: xr.DataArray,
        admin_mask: np.ndarray,
        all_cropland_masks: dict,
        var_name: str,
        lat_coord: str,
        lon_coord: str,
    ) -> pd.DataFrame:
        """Vectorized aggregation for all admin boundaries simultaneously

        This replaces the slow boundary-by-boundary loop with efficient numpy operations
        that process all admin units at once using combined masks.

        Args:
            var_data: Weather data to aggregate
            admin_mask: Admin boundary assignments (2D array)
            all_cropland_masks: Dictionary of cropland masks for each year
            var_name: Name of the weather variable
            lat_coord: Latitude coordinate name
            lon_coord: Longitude coordinate name

        Returns:
            DataFrame with aggregated results for all admin units
        """
        # Find admin units that have cropland in ANY year
        union_valid_mask = np.zeros_like(admin_mask, dtype=bool)
        for year_mask in all_cropland_masks.values():
            union_valid_mask |= (admin_mask >= 0) & year_mask

        valid_admin_indices = np.unique(admin_mask[union_valid_mask])
        logger.info(
            f"Processing {len(valid_admin_indices)} admin units with cropland (filtered from {len(self.boundaries)} total)"
        )

        # Convert coverage weights to numpy array for vectorized operations
        # Shape: (n_admin_units, n_lats, n_lons)
        if self.coverage_weights is None:
            raise ValueError("Coverage weights not computed")
        weights_array = np.stack(self.coverage_weights, axis=0)

        results = []

        # Group time steps by year for efficient processing (cropland doesn't change daily)
        time_values = var_data.time.values
        years_to_indices = {}
        for time_idx, time_val in enumerate(time_values):
            year = pd.to_datetime(time_val).year
            if year not in years_to_indices:
                years_to_indices[year] = []
            years_to_indices[year].append(time_idx)

        logger.info(
            f"Processing {len(years_to_indices)} years with vectorized operations..."
        )

        # Process each year with its cropland mask (much more efficient)
        for year in tqdm(sorted(years_to_indices.keys()), desc="Processing years"):
            # Get the appropriate cropland mask for this year
            if year in all_cropland_masks:
                cropland_mask = all_cropland_masks[year]
            else:
                # Use nearest available year if exact year not found
                available_years = list(all_cropland_masks.keys())
                nearest_year = min(available_years, key=lambda x: abs(x - year))
                cropland_mask = all_cropland_masks[nearest_year]

            # Get all time indices for this year
            year_time_indices = years_to_indices[year]

            # Process all days of this year at once with the same cropland mask
            year_data = var_data.isel(
                time=year_time_indices
            )  # Shape: (n_days, n_lats, n_lons)
            year_time_values = [time_values[i] for i in year_time_indices]

            # Vectorized aggregation for all admin units and all days of this year
            year_results = self._vectorized_weighted_average_all_admins_year(
                year_data,
                admin_mask,
                cropland_mask,
                weights_array,
                valid_admin_indices,
            )

            # Create DataFrame for all days of this year
            for day_idx, time_val in enumerate(year_time_values):
                time_df = pd.DataFrame(
                    {
                        "time": time_val,
                        "admin_id": valid_admin_indices,
                        var_name: year_results[day_idx],  # Results for this day
                    }
                )
                results.append(time_df)

        # Combine all time slices
        combined_df = pd.concat(results, ignore_index=True)

        # Add admin hierarchy info from GADM
        admin_info = []
        for admin_idx in valid_admin_indices:
            boundary = self.boundaries.iloc[admin_idx]
            centroid = boundary.geometry.centroid
            admin_info.append(
                {
                    "admin_id": admin_idx,
                    "country_name": boundary["COUNTRY"],
                    "admin_level_1_name": boundary["NAME_1"],
                    "admin_level_2_name": boundary["NAME_2"],
                    "latitude": round(centroid.y, 3),
                    "longitude": round(centroid.x, 3),
                }
            )

        admin_info_df = pd.DataFrame(admin_info)

        # Merge with aggregated data
        result_df = combined_df.merge(admin_info_df, on="admin_id", how="left")

        return result_df

    def _vectorized_weighted_average_all_admins_year(
        self,
        year_data: xr.DataArray,
        admin_mask: np.ndarray,
        cropland_mask: np.ndarray,
        weights_array: np.ndarray,
        valid_admin_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute weighted averages for all admin units for all days of a year

        This is the most efficient approach: process all days of a year together
        since the cropland mask doesn't change within a year.

        Args:
            year_data: Weather data for all days of one year (n_days, n_lats, n_lons)
            admin_mask: Admin boundary assignments (n_lats, n_lons)
            cropland_mask: Cropland mask for this year (n_lats, n_lons)
            weights_array: Coverage weights for all admin units (n_admin_units, n_lats, n_lons)
            valid_admin_indices: Admin unit indices to process

        Returns:
            Array of weighted averages (n_days, n_admin_units)
        """
        n_days = year_data.shape[0]
        n_admins = len(valid_admin_indices)
        results = np.full((n_days, n_admins), np.nan)

        # Convert year_data to numpy for faster processing
        year_data_values = year_data.values  # Shape: (n_days, n_lats, n_lons)

        for i, admin_idx in enumerate(valid_admin_indices):
            # Create combined mask for this admin unit
            admin_cells = admin_mask == admin_idx
            valid_cells = admin_cells & cropland_mask

            if not np.any(valid_cells):
                continue  # No valid cells for this admin unit

            # Get weights for this admin unit
            admin_weights = weights_array[admin_idx]

            # Apply combined filtering
            valid_weights = admin_weights * valid_cells

            # Skip if no valid weights
            if not np.any(valid_weights > 0):
                continue

            # Compute weighted average for all days at once using broadcasting
            # year_data_values: (n_days, n_lats, n_lons)
            # valid_weights: (n_lats, n_lons)

            # Create mask for valid (non-NaN) weather data
            valid_data_mask = ~np.isnan(year_data_values)

            # Apply weights only where we have valid weather data
            # This ensures NaN weather data doesn't contribute to numerator or denominator
            valid_weights_expanded = valid_weights[None, :, :] * valid_data_mask

            # Compute weighted sum (numerator) - use nansum to handle NaN values properly
            weighted_data = year_data_values * valid_weights_expanded
            weighted_sum = np.nansum(weighted_data, axis=(1, 2))

            # Compute sum of weights (denominator) - only count weights where data is valid
            weight_sum = np.sum(valid_weights_expanded, axis=(1, 2))

            # Compute weighted average only where we have valid weights
            valid_days = weight_sum > 0
            results[valid_days, i] = weighted_sum[valid_days] / weight_sum[valid_days]

        return results

    def _load_cropland_data(self):
        """Load HYDE-3.5 cropland data"""
        cropland_path = Path("data/global/hyde-3.5/cropland.nc")
        self.cropland_data = xr.open_dataset(cropland_path)
        logger.info(f"Loaded cropland data")

    def _extract_all_years_from_dataset(self, dataset: xr.Dataset) -> List[int]:
        """Extract all years from weather dataset time dimension."""
        data_vars = [
            var for var in dataset.data_vars if var not in ["crs", "spatial_ref"]
        ]
        var_data = dataset[data_vars[0]]
        all_times = var_data.time.values
        years = [pd.to_datetime(t).year for t in all_times]
        return years

    def _get_or_compute_admin_mask(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """FILTERING: Assign each grid cell to an admin boundary using vectorized rasterization

        Creates a 2D array where each grid cell contains the admin unit index it belongs to,
        or -1 if outside any boundary. Uses fast vectorized rasterization instead of
        slow point-in-polygon loops.

        Cached globally since admin boundaries don't change.
        """
        cache_file = (
            self.cropland_mask_dir / f"admin_mask_{self.country_param.lower()}.pkl"
        )

        if self.admin_mask_cache is not None:
            return self.admin_mask_cache

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                self.admin_mask_cache = cached_data["admin_mask"]
                return self.admin_mask_cache

        logger.info("Computing admin boundary mask using vectorized rasterization...")

        # Create raster transform for the grid
        lat_step = abs(lats[1] - lats[0])
        lon_step = abs(lons[1] - lons[0])

        transform = from_bounds(
            lons.min() - lon_step / 2,
            lats.min() - lat_step / 2,
            lons.max() + lon_step / 2,
            lats.max() + lat_step / 2,
            len(lons),
            len(lats),
        )

        # Prepare geometries with admin indices
        geometries = [
            (geom, admin_idx) for admin_idx, geom in enumerate(self.boundaries.geometry)
        ]

        # Use rasterio to efficiently rasterize all boundaries at once
        admin_mask = rasterio.features.rasterize(
            geometries,
            out_shape=(len(lats), len(lons)),
            transform=transform,
            fill=-1,  # Fill value for areas outside any boundary
            dtype=np.int32,
        )

        # Cache the result
        cache_data = {"admin_mask": admin_mask, "lats": lats, "lons": lons}
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        self.admin_mask_cache = admin_mask
        logger.info(
            f"Computed and cached admin boundary mask for {len(self.boundaries)} boundaries"
        )

        return admin_mask

    def _get_or_compute_all_cropland_masks(
        self, lats: np.ndarray, lons: np.ndarray, years: List[int]
    ) -> dict:
        """FILTERING: Pre-compute cropland masks for all years and cache them together"""
        # Check if we have a combined cache file for all years
        years_str = f"{min(years)}_{max(years)}"
        combined_cache_file = (
            self.cropland_mask_dir / f"all_cropland_masks_{years_str}.pkl"
        )

        if combined_cache_file.exists():
            logger.info(
                f"Loading cached cropland masks for years {min(years)}-{max(years)}"
            )
            with open(combined_cache_file, "rb") as f:
                cached_data = pickle.load(f)
                return cached_data["all_cropland_masks"]

        logger.info(
            f"Computing cropland masks for {len(set(years))} unique years ({min(years)}-{max(years)})..."
        )

        all_cropland_masks = {}
        unique_years = sorted(set(years))  # Remove duplicates and sort

        for year in tqdm(unique_years, desc="Computing cropland masks"):
            all_cropland_masks[year] = self._compute_cropland_mask_for_year(
                lats, lons, year
            )

        # Cache all masks together
        cache_data = {
            "all_cropland_masks": all_cropland_masks,
            "lats": lats,
            "lons": lons,
            "years": unique_years,
        }
        with open(combined_cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Computed and cached cropland masks for all years")
        return all_cropland_masks

    def _compute_cropland_mask_for_year(
        self, lats: np.ndarray, lons: np.ndarray, year: int
    ) -> np.ndarray:
        """Compute cropland mask for a specific year (without individual caching)"""
        # Get cropland data for this year
        target_time = cftime.DatetimeNoLeap(year, 6, 1)
        cropland_year_data = self.cropland_data.sel(time=target_time, method="nearest")

        # Interpolate cropland data to weather grid using xarray's built-in interpolation
        cropland_interp = cropland_year_data.cropland.interp(
            lat=lats, lon=lons, method="nearest"
        )

        # Create boolean mask for areas with cropland
        cropland_mask = cropland_interp.values > 0
        return cropland_mask

    def _get_or_compute_coverage_weights(self, transform, out_shape, lats, lons):
        """AVERAGING: Compute area-weighted coverage for each admin boundary

        For each admin boundary, creates a 2D array where each grid cell contains
        the fraction of that cell covered by the boundary (0.0 to 1.0).

        Uses 25-subcell subdivision: divides each 0.1° grid cell into 5x5 subcells,
        counts how many subcells fall within the boundary, and uses that fraction
        as the coverage weight for proper area-weighted averaging.

        Cached globally since boundaries don't change.
        """
        cache_file = (
            self.cropland_mask_dir
            / f"coverage_weights_{self.country_param.lower()}.pkl"
        )

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info("Computing coverage weights with 25 subcells per grid cell...")
        coverage_weights = self._compute_coverage_weights(
            transform, out_shape, lats, lons
        )

        with open(cache_file, "wb") as f:
            pickle.dump(coverage_weights, f)

        return coverage_weights

    def _compute_coverage_weights(self, transform, out_shape, lats, lons):
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
                    cell_lat_min = grid_lat - lat_step / 2
                    cell_lat_max = grid_lat + lat_step / 2
                    cell_lon_min = grid_lon - lon_step / 2
                    cell_lon_max = grid_lon + lon_step / 2

                    covered_subcells = 0
                    total_subcells = 25

                    # Check each of the 25 subcells
                    for sub_i in range(5):
                        for sub_j in range(5):
                            # Subcell center
                            sub_lat = cell_lat_min + (sub_i + 0.5) * lat_step / 5
                            sub_lon = cell_lon_min + (sub_j + 0.5) * lon_step / 5

                            # Create point for subcell center
                            subcell_point = Point(sub_lon, sub_lat)

                            if geom.contains(subcell_point):
                                covered_subcells += 1

                    # Coverage fraction for this grid cell
                    coverage = covered_subcells / total_subcells
                    weights[i, j] = coverage

            all_weights.append(weights)

        return all_weights

    def aggregate_file(self, file_path: str) -> pd.DataFrame:
        """Aggregate a single NetCDF file to administrative boundaries"""
        logger.info(f"Processing file: {file_path}")

        dataset = xr.open_dataset(file_path)
        result = self.aggregate_dataset(dataset)

        return result
