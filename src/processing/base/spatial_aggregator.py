"""Spatial aggregator for averaging gridded data over administrative boundaries"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path
import cftime
import datetime

import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
import rasterio.features
from rasterio.transform import from_bounds

from src.processing.base.processor import BaseProcessor
from src.constants import CHUNK_SIZE_TIME_PROCESSING

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
        # Ensure consistent indexing for admin ID lookups
        self.boundaries_indexed = self.boundaries.reset_index(drop=True)
        self.country_param = country_param

        # Set up cache directories under processed folder
        processed_dir = Path("data") / country_param.lower() / "processed"
        self.cropland_mask_dir = processed_dir / "cropland_mask"
        self.admin_mask_dir = processed_dir / "admin_mask"
        self.weather_processed_dir = processed_dir / "weather"

        self.cropland_mask_dir.mkdir(parents=True, exist_ok=True)
        self.admin_mask_dir.mkdir(parents=True, exist_ok=True)
        self.weather_processed_dir.mkdir(parents=True, exist_ok=True)

        # Load HYDE cropland data once
        self._load_cropland_data()

        # Precompute boundary centroids to avoid repeated calculations
        # Convert to projected CRS for accurate centroids, then back to geographic
        projected_boundaries = self.boundaries_indexed.to_crs(
            "EPSG:3857"
        )  # Web Mercator
        centroids_projected = projected_boundaries.geometry.centroid

        # Ensure we have a valid CRS before converting back
        original_crs = self.boundaries_indexed.crs
        if original_crs is None:
            # Default to WGS84 if no CRS is set
            original_crs = "EPSG:4326"

        # Reset index to ensure admin_id matches position for centroid lookup
        self.boundary_centroids = centroids_projected.to_crs(original_crs).reset_index(
            drop=True
        )

        logger.info(
            f"SpatialAggregator initialized for {country_param} with {len(self.boundaries_indexed)} admin boundaries"
        )

    def aggregate_dataset(self, dataset: xr.Dataset) -> pd.DataFrame:
        """Aggregate xarray dataset to administrative boundaries using weighted averaging

        Args:
            dataset: xarray Dataset with spatial data (daily or weekly weather data)

        Returns:
            DataFrame with time, admin_name, admin_id, value columns
        """
        # Extract dataset info
        all_years = self._extract_all_years_from_dataset(dataset)
        var_data = self._get_main_data_variable(dataset)
        lats, lons, lat_step, lon_step = self._extract_coordinates(var_data)

        logger.info(f"Dataset spans years {min(all_years)} to {max(all_years)}")

        # Load or create masks
        admin_mask = self._load_or_create_admin_mask(lats, lons, lat_step, lon_step)
        cropland_masks = self._load_or_create_cropland_masks(all_years, lats, lons)

        # Check if this is weekly aggregated data
        if "time" in dataset.dims:
            # Daily data processing
            results = self._process_time_series(
                var_data, admin_mask, cropland_masks, lats, lons, lat_step, lon_step
            )
        elif "week" in dataset.dims:
            # Weekly aggregated data processing
            results = self._process_weekly_data(var_data, admin_mask, cropland_masks)
        else:
            raise ValueError(
                "Dataset must have either 'time' or 'week' dimension for processing"
            )

        df = pd.DataFrame(results)
        logger.debug(f"Spatial aggregation complete: {len(df)} records generated")
        return df

    def _get_main_data_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        """Extract the main data variable from dataset"""
        data_vars = [
            var for var in dataset.data_vars if var not in ["crs", "spatial_ref"]
        ]
        assert (
            len(data_vars) == 1
        ), f"Expected 1 data variable, got {len(data_vars)}: {data_vars}"
        return dataset[data_vars[0]]

    def _extract_coordinates(
        self, var_data: xr.DataArray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Extract coordinate arrays and step sizes"""
        lats = var_data["lat"].values
        lons = var_data["lon"].values
        lat_step = abs(lats[1] - lats[0])
        lon_step = abs(lons[1] - lons[0])
        return lats, lons, lat_step, lon_step

    def _cache_mask_as_netcdf(
        self,
        mask: np.ndarray,
        file_path: Path,
        lats: np.ndarray,
        lons: np.ndarray,
        mask_name: str,
    ):
        """Generic helper to cache mask as NetCDF"""
        mask_da = xr.DataArray(
            mask,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            name=mask_name,
        )
        mask_da.to_netcdf(file_path)

    def _load_mask_from_netcdf(self, file_path: Path, mask_name: str) -> np.ndarray:
        """Generic helper to load mask from NetCDF"""
        with xr.open_dataset(file_path) as ds:
            return ds[mask_name].values

    def _load_or_create_admin_mask(
        self, lats: np.ndarray, lons: np.ndarray, lat_step: float, lon_step: float
    ) -> np.ndarray:
        """Load cached admin mask or create new one"""
        admin_mask_file = self.admin_mask_dir / "admin_mask.nc"
        area_weights_file = self.admin_mask_dir / "area_weights.npy"

        if admin_mask_file.exists() and area_weights_file.exists():
            logger.info("Loading cached admin boundary mask...")
            admin_mask = self._load_mask_from_netcdf(admin_mask_file, "admin_mask")
            self.area_weights = np.load(area_weights_file)
            return admin_mask
        else:
            logger.info("Creating admin boundary mask (first time for this grid)...")
            admin_mask = self._create_admin_mask(lats, lons, lat_step, lon_step)
            self._cache_mask_as_netcdf(
                admin_mask, admin_mask_file, lats, lons, "admin_mask"
            )
            # Save area weights
            np.save(area_weights_file, self.area_weights)
            logger.info(f"Admin mask created and cached: {admin_mask.shape}")
            return admin_mask

    def _load_or_create_cropland_masks(
        self, all_years: List[int], lats: np.ndarray, lons: np.ndarray
    ) -> dict:
        """Load or create cropland masks for all years"""
        cropland_masks = {}
        for year in all_years:
            cropland_mask_file = self.cropland_mask_dir / f"cropland_mask_{year}.nc"

            if cropland_mask_file.exists():
                logger.debug(f"Loading cached cropland mask for {year}...")
                cropland_masks[year] = self._load_mask_from_netcdf(
                    cropland_mask_file, "cropland_mask"
                )
            else:
                logger.info(f"Creating cropland mask for {year}...")
                cropland_mask = self._create_cropland_mask(lats, lons, year)
                cropland_masks[year] = cropland_mask
                self._cache_mask_as_netcdf(
                    cropland_mask, cropland_mask_file, lats, lons, "cropland_mask"
                )
                logger.debug(f"Cached cropland mask for {year}: {cropland_mask_file}")

        return cropland_masks

    def _process_time_series(
        self,
        var_data: xr.DataArray,
        admin_mask: np.ndarray,
        cropland_masks: dict,
        lats: np.ndarray,
        lons: np.ndarray,
        lat_step: float,
        lon_step: float,
    ) -> List[dict]:
        """Process time series data in chunks"""
        logger.info("Computing area-weighted averages with filtering...")

        results = []
        time_values = var_data.time.values

        # Process in chunks to manage memory
        total_chunks = (
            len(time_values) + CHUNK_SIZE_TIME_PROCESSING - 1
        ) // CHUNK_SIZE_TIME_PROCESSING

        for chunk_idx in tqdm(range(total_chunks), desc="Processing time chunks"):
            start_idx = chunk_idx * CHUNK_SIZE_TIME_PROCESSING
            end_idx = min(start_idx + CHUNK_SIZE_TIME_PROCESSING, len(time_values))

            chunk_times = time_values[start_idx:end_idx]
            chunk_data = var_data.isel(time=slice(start_idx, end_idx))

            chunk_results = self._process_time_chunk(
                chunk_data,
                chunk_times,
                admin_mask,
                cropland_masks,
                lats,
                lons,
                lat_step,
                lon_step,
            )
            results.extend(chunk_results)

        return results

    def _process_weekly_data(
        self, var_data: xr.DataArray, admin_mask: np.ndarray, cropland_masks: dict
    ) -> List[dict]:
        """Process weekly aggregated data"""
        logger.debug("Processing weekly aggregated data...")
        results = []

        # Get unique years from the year coordinate
        years = np.unique(var_data.year.values)

        # Iterate over weeks (the main dimension)
        for week_idx, week in tqdm(
            enumerate(var_data.week.values),
            total=len(var_data.week.values),
            desc="Processing weeks",
        ):
            week_int = int(week)
            year_int = int(var_data.year.values[week_idx])

            cropland_mask = cropland_masks.get(
                year_int, np.ones_like(admin_mask, dtype=bool)
            )

            # Get data slice for this week
            data_slice = var_data.isel(week=week_idx).values

            # Skip if all NaN
            if np.isnan(data_slice).all():
                continue

            # Get combined mask for this year
            combined_mask = (admin_mask >= 0) & cropland_mask

            # Create time string - just use year-01-01 + week offset, simple and clean
            jan1 = datetime.date(year_int, 1, 1)
            week_start = jan1 + datetime.timedelta(weeks=week_int - 1)
            time_str = week_start.strftime("%Y-%m-%d")

            # Vectorized area-weighted averaging for all admin units at once
            admin_results = self._compute_vectorized_averages_by_year(
                data_slice, admin_mask, combined_mask, year_int, time_str
            )
            results.extend(admin_results)

        return results

    def _extract_all_years_from_dataset(self, dataset: xr.Dataset) -> List[int]:
        """Extract all unique years from the dataset"""
        if "time" in dataset.dims:
            # Daily data with time dimension
            time_values = dataset.time.values
            years = set()
            for time_val in time_values:
                if isinstance(time_val, cftime.datetime):
                    years.add(time_val.year)
                elif hasattr(time_val, "year"):
                    years.add(time_val.year)
                else:
                    dt = pd.to_datetime(str(time_val))
                    years.add(dt.year)
        elif "year" in dataset.dims:
            # Weekly aggregated data with year dimension
            years = set(dataset.year.values)
        elif "year" in dataset.coords:
            # Weekly aggregated data with year coordinate
            years = set(np.unique(dataset.year.values))
        else:
            raise ValueError(
                "Dataset must have either 'time' dimension or 'year' dimension/coordinate"
            )

        return sorted(list(years))

    def _load_cropland_data(self):
        """Load HYDE cropland data"""
        logger.info("Loading HYDE cropland data...")

        hyde_file = Path("data/global/hyde-3.5/cropland.nc")

        if not hyde_file.exists():
            logger.warning("HYDE data not available - skipping cropland filtering")
            self.hyde_data = None
            return

        self.hyde_data = xr.open_dataset(hyde_file)
        logger.info("HYDE cropland data loaded successfully")

    def _create_admin_mask(
        self, lats: np.ndarray, lons: np.ndarray, lat_step: float, lon_step: float
    ) -> np.ndarray:
        """Create admin boundary mask for the grid with 25-subcell area weighting"""
        # Create 5x5 subdivision for each grid cell (25 subcells total)
        subcell_factor = 5
        high_res_lats = np.linspace(
            lats.min() - lat_step / 2,
            lats.max() + lat_step / 2,
            len(lats) * subcell_factor + 1,
        )[:-1] + lat_step / (2 * subcell_factor)
        high_res_lons = np.linspace(
            lons.min() - lon_step / 2,
            lons.max() + lon_step / 2,
            len(lons) * subcell_factor + 1,
        )[:-1] + lon_step / (2 * subcell_factor)

        # Create transform for high-resolution rasterization
        transform = from_bounds(
            high_res_lons.min() - lon_step / (2 * subcell_factor),
            high_res_lats.min() - lat_step / (2 * subcell_factor),
            high_res_lons.max() + lon_step / (2 * subcell_factor),
            high_res_lats.max() + lat_step / (2 * subcell_factor),
            len(high_res_lons),
            len(high_res_lats),
        )

        # Rasterize admin boundaries at high resolution
        shapes = [
            (geom, idx) for idx, geom in enumerate(self.boundaries_indexed.geometry)
        ]
        high_res_mask = rasterio.features.rasterize(
            shapes,
            out_shape=(len(high_res_lats), len(high_res_lons)),
            transform=transform,
            fill=-1,
        )

        # Vectorized downsampling to original resolution
        admin_mask = np.full((len(lats), len(lons)), -1, dtype=np.int32)
        area_weights = np.zeros(
            (len(lats), len(lons), len(self.boundaries_indexed)), dtype=np.float32
        )

        # Reshape high-res mask into blocks for vectorized processing
        high_res_reshaped = (
            high_res_mask.reshape(len(lats), subcell_factor, len(lons), subcell_factor)
            .transpose(0, 2, 1, 3)
            .reshape(len(lats), len(lons), subcell_factor * subcell_factor)
        )

        # Vectorized computation of area weights and dominant admin
        for i in range(len(lats)):
            for j in range(len(lons)):
                subcell_block = high_res_reshaped[i, j, :]

                # Count admin IDs in this block (excluding -1)
                valid_mask = subcell_block >= 0
                if valid_mask.any():
                    unique_ids, counts = np.unique(
                        subcell_block[valid_mask], return_counts=True
                    )
                    # Store area weights for each admin unit
                    area_weights[i, j, unique_ids] = counts / 25.0  # 25 subcells total

                    # Assign the admin ID with the largest area
                    dominant_admin = unique_ids[np.argmax(counts)]
                    admin_mask[i, j] = dominant_admin

        # Store area weights for later use
        self.area_weights = area_weights
        return admin_mask

    def _create_cropland_mask(
        self, lats: np.ndarray, lons: np.ndarray, year: int
    ) -> np.ndarray:
        """Create cropland mask for the given year and grid"""
        if self.hyde_data is None:
            # If no HYDE data, return all ones (no filtering)
            return np.ones((len(lats), len(lons)), dtype=bool)

        # Find the closest year in HYDE data
        hyde_years = self.hyde_data.time.dt.year.values
        closest_year_idx = np.argmin(np.abs(hyde_years - year))
        closest_year = hyde_years[closest_year_idx]

        logger.debug(f"Using HYDE year {closest_year} for requested year {year}")

        # Extract cropland data for the year
        cropland_data = self.hyde_data.isel(time=closest_year_idx)

        # Get the main cropland variable (assuming first data variable)
        data_vars = [
            var for var in cropland_data.data_vars if var not in ["crs", "spatial_ref"]
        ]
        cropland_var = cropland_data[data_vars[0]]

        # Interpolate to our grid
        target_coords = {"lat": lats, "lon": lons}
        interpolated = cropland_var.interp(target_coords, method="linear")

        # Create mask: areas with significant cropland (>1% coverage)
        cropland_fraction = interpolated.values
        cropland_mask = cropland_fraction > 0.01  # 1% threshold

        return cropland_mask

    def _process_time_chunk(
        self,
        chunk_data: xr.DataArray,
        chunk_times: np.ndarray,
        admin_mask: np.ndarray,
        cropland_masks: dict,
        lats: np.ndarray,
        lons: np.ndarray,
        lat_step: float,
        lon_step: float,
    ) -> List[dict]:
        """Process a chunk of time data with per-year optimization"""
        results = []

        # Group times by year for efficient processing
        year_groups = {}
        time_year_mapping = {}

        for time_idx, time_val in enumerate(chunk_times):
            year, time_str = self._extract_year_and_time_str(time_val)
            time_year_mapping[time_idx] = (year, time_str)

            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(time_idx)

        # Process per year instead of per admin
        for year, time_indices in year_groups.items():
            cropland_mask = cropland_masks.get(
                year, np.ones_like(admin_mask, dtype=bool)
            )

            # Process all time steps for this year at once
            for time_idx in time_indices:
                year, time_str = time_year_mapping[time_idx]
                data_slice = chunk_data.isel(time=time_idx).values

                # Skip if all NaN
                if np.isnan(data_slice).all():
                    continue

                # Get combined mask for this year
                combined_mask = (admin_mask >= 0) & cropland_mask

                # Vectorized area-weighted averaging for all admin units at once
                admin_results = self._compute_vectorized_averages_by_year(
                    data_slice, admin_mask, combined_mask, year, time_str
                )
                results.extend(admin_results)

        return results

    def _extract_year_and_time_str(self, time_val) -> Tuple[int, str]:
        """Extract year and formatted time string from time value"""
        if isinstance(time_val, cftime.datetime):
            return (
                time_val.year,
                f"{time_val.year}-{time_val.month:02d}-{time_val.day:02d}",
            )
        elif hasattr(time_val, "year"):
            return time_val.year, str(time_val)[:10]
        else:
            dt = pd.to_datetime(str(time_val))
            return dt.year, dt.strftime("%Y-%m-%d")

    def _compute_vectorized_averages_by_year(
        self,
        data_slice: np.ndarray,
        admin_mask: np.ndarray,
        combined_mask: np.ndarray,
        year: int,
        time_str: str,
    ) -> List[dict]:
        """Compute area-weighted averages for all admin units using full vectorization"""
        results = []

        # Get unique admin IDs that have valid data
        valid_admin_ids = np.unique(admin_mask[combined_mask & ~np.isnan(data_slice)])
        valid_admin_ids = valid_admin_ids[valid_admin_ids >= 0]  # Remove -1 (no admin)

        if len(valid_admin_ids) == 0:
            return results

        # Create master mask for valid data
        master_valid_mask = combined_mask & ~np.isnan(data_slice)

        # Extract all relevant area weights for valid admin IDs
        relevant_weights = self.area_weights[
            :, :, valid_admin_ids
        ]  # Shape: (lat, lon, num_valid_admins)

        # Apply master mask to weights and data
        masked_weights = relevant_weights * master_valid_mask[:, :, np.newaxis]
        masked_data = np.where(master_valid_mask, data_slice, 0)

        # Vectorized computation for all admin units at once
        # Sum over spatial dimensions to get totals per admin
        weighted_sums = np.nansum(
            masked_data[:, :, np.newaxis] * masked_weights, axis=(0, 1)
        )  # Shape: (num_valid_admins,)
        total_weights = np.nansum(
            masked_weights, axis=(0, 1)
        )  # Shape: (num_valid_admins,)

        # Compute averages where weights > 0
        valid_weight_mask = total_weights > 0
        avg_values = weighted_sums / np.where(total_weights > 0, total_weights, 1)

        # Create results for admin units with valid averages
        valid_indices = np.where(valid_weight_mask)[0]

        for idx in valid_indices:
            admin_id = valid_admin_ids[idx]
            avg_value = avg_values[idx]

            # Get admin info using precomputed centroids
            admin_row = self.boundaries_indexed.iloc[int(admin_id)]
            admin_name = self.base_processor.get_admin_name(admin_row)
            centroid = self.boundary_centroids.iloc[int(admin_id)]

            result_dict = {
                "time": time_str,
                "admin_name": admin_name,
                "admin_id": int(admin_id),
                "value": float(avg_value),
                "country_name": self.base_processor.country_full_name,
                "latitude": centroid.y,
                "longitude": centroid.x,
            }

            # Add dynamic admin level columns based on target admin level
            for level in range(1, self.base_processor.admin_level + 1):
                result_dict[f"admin_level_{level}_name"] = admin_row.get(
                    f"NAME_{level}", ""
                )

            results.append(result_dict)

        return results

    def aggregate_file(self, file_path: str) -> pd.DataFrame:
        """Aggregate a single NetCDF file to administrative boundaries"""
        logger.debug(f"Aggregating file: {file_path}")

        with xr.open_dataset(file_path) as dataset:
            return self.aggregate_dataset(dataset)
