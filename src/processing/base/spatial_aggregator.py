"""Spatial aggregator for averaging gridded data over administrative boundaries"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path
import datetime

import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
import rasterio.features
from rasterio.transform import from_bounds

from src.processing.base.processor import BaseProcessor
from src.constants import SUBCELL_SIZE

logger = logging.getLogger(__name__)


class SpatialAggregator:
    """Aggregates weekly spatial data to administrative boundaries with cropland filtering

    Optimized for weekly aggregated data only. Two-stage process:
    1. FILTERING: Create masks to identify which grid cells to process
       - Admin boundary mask: Which admin unit each grid cell belongs to (cached globally)
       - Cropland mask: Which grid cells have cropland (cached per year)
       - Combined: Only process admin units that have cropland

    2. AVERAGING: For each admin unit, compute area-weighted average
       - Uses SUBCELL_SIZE-subcell subdivision per grid cell for proper area weighting
       - Combines area weights with cropland filtering for final aggregation
       - Results are cached to avoid recomputation
    """

    def __init__(
        self, base_processor: BaseProcessor, country_param: str, cropland_filter: bool
    ):
        """Initialize with a base processor that provides boundaries

        Args:
            base_processor: Provides admin boundaries and country info
            country_param: Country parameter for processing
            cropland_filter: If True, filter aggregation to cropland areas only.
                           If False, aggregate over all valid data areas.
        """
        self.base_processor = base_processor
        self.boundaries = base_processor.boundaries
        # Ensure consistent indexing for admin ID lookups
        self.boundaries_indexed = self.boundaries.reset_index(drop=True)
        self.country_param = country_param
        self.cropland_filter = cropland_filter

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

        # Pre-compute admin info arrays for fast access (optimization #1)
        self._precompute_admin_arrays()

        logger.info(
            f"SpatialAggregator initialized for {country_param} with {len(self.boundaries_indexed)} admin boundaries"
        )

    def aggregate_dataset(
        self, dataset: xr.Dataset, variable_name: str
    ) -> pd.DataFrame:
        """Aggregate xarray dataset to administrative boundaries using weighted averaging

        Args:
            dataset: xarray Dataset with weekly aggregated spatial data
            variable_name: Name of variable for checkpointing

        Returns:
            DataFrame with time, admin_name, admin_id, value columns
        """
        # Check for cached results year by year
        all_years = self._extract_all_years_from_dataset(dataset)

        # Check if all years are cached
        all_cached = True
        cached_dfs = []

        for year in all_years:
            cache_file = self._get_cache_filename(variable_name, year)
            if cache_file.exists():
                logger.debug(f"Loading cached result for {variable_name} {year}")
                cached_dfs.append(pd.read_csv(cache_file))
            else:
                all_cached = False
                break

        if all_cached:
            logger.info(
                f"Loading cached spatial aggregation results for {variable_name} ({len(all_years)} years)"
            )
            return pd.concat(cached_dfs, ignore_index=True)

        logger.info(
            f"Computing spatial aggregation for {variable_name} ({len(all_years)} years, will cache by year)"
        )

        # Extract dataset info
        var_data = self._get_main_data_variable(dataset)
        lats, lons, lat_step, lon_step = self._extract_coordinates(var_data)

        logger.info(f"Dataset spans years {min(all_years)} to {max(all_years)}")

        # Load or create masks
        admin_mask = self._load_or_create_admin_mask(lats, lons, lat_step, lon_step)
        cropland_masks = {}
        if self.cropland_filter:
            cropland_masks = self._load_or_create_cropland_masks(all_years, lats, lons)

        # Process weekly aggregated data
        if "week" in dataset.dims:
            results = self._process_weekly_data(var_data, admin_mask, cropland_masks)
        else:
            raise ValueError(
                "Dataset must have 'week' dimension for processing. Daily data should be temporally aggregated first."
            )

        df = pd.DataFrame(results)

        # Cache the results year by year
        for year in all_years:
            year_df = df[df["time"].str.startswith(str(year))]
            if not year_df.empty:
                cache_file = self._get_cache_filename(variable_name, year)
                year_df.to_csv(cache_file, index=False)
                logger.debug(f"Cached {year} results to {cache_file}")

        logger.debug(f"Spatial aggregation complete: {len(df)} records generated")
        return df

    def _get_cache_filename(self, variable_name: str, year: int) -> Path:
        """Generate cache filename for spatial aggregation results"""
        filename = f"{year}_{variable_name}_weekly_weighted_admin{self.base_processor.admin_level}.csv"
        return self.weather_processed_dir / filename

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
        # Include grid dimensions in cache filename to avoid conflicts
        grid_shape = f"{len(lats)}x{len(lons)}"
        admin_mask_file = self.admin_mask_dir / f"admin_mask_{grid_shape}.nc"
        area_weights_file = self.admin_mask_dir / f"area_weights_{grid_shape}.npy"

        if admin_mask_file.exists() and area_weights_file.exists():
            logger.info(f"Loading cached admin boundary mask for grid {grid_shape}...")
            admin_mask = self._load_mask_from_netcdf(admin_mask_file, "admin_mask")
            self.area_weights = np.load(area_weights_file)
            return admin_mask
        else:
            logger.info(
                f"Creating admin boundary mask for grid {grid_shape} (first time)..."
            )
            admin_mask = self._create_admin_mask(lats, lons, lat_step, lon_step)
            self._cache_mask_as_netcdf(
                admin_mask, admin_mask_file, lats, lons, "admin_mask"
            )
            # Save area weights
            np.save(area_weights_file, self.area_weights)
            logger.info(
                f"Admin mask created and cached for grid {grid_shape}: {admin_mask.shape}"
            )
            return admin_mask

    def _load_or_create_cropland_masks(
        self, all_years: List[int], lats: np.ndarray, lons: np.ndarray
    ) -> dict:
        """Load or create cropland masks for all years"""
        cropland_masks = {}
        # Include grid dimensions in cache filename to avoid conflicts
        grid_shape = f"{len(lats)}x{len(lons)}"
        logger.info(f"Creating cropland masks for {len(all_years)} years...")
        for year in all_years:
            cropland_mask_file = (
                self.cropland_mask_dir / f"cropland_mask_{year}_{grid_shape}.nc"
            )

            if cropland_mask_file.exists():
                logger.debug(
                    f"Loading cached cropland mask for {year} (grid {grid_shape})..."
                )
                cropland_masks[year] = self._load_mask_from_netcdf(
                    cropland_mask_file, "cropland_mask"
                )
            else:
                cropland_mask = self._create_cropland_mask(lats, lons, year)
                cropland_masks[year] = cropland_mask
                self._cache_mask_as_netcdf(
                    cropland_mask, cropland_mask_file, lats, lons, "cropland_mask"
                )
                logger.debug(
                    f"Cached cropland mask for {year} (grid {grid_shape}): {cropland_mask_file}"
                )

        return cropland_masks

    def _process_weekly_data(
        self, var_data: xr.DataArray, admin_mask: np.ndarray, cropland_masks: dict
    ) -> List[dict]:
        """Process weekly aggregated data with full vectorization"""
        logger.debug("Processing weekly aggregated data...")
        results = []

        # Group weeks by year for efficient batch processing
        year_to_weeks = {}
        for week_idx, (week, year) in enumerate(
            zip(var_data.week.values, var_data.year.values)
        ):
            year_int = int(year)
            if year_int not in year_to_weeks:
                year_to_weeks[year_int] = []
            year_to_weeks[year_int].append((week_idx, int(week), year_int))

        # Process all weeks for each year in batches
        for year_int, week_info_list in tqdm(
            year_to_weeks.items(), desc="Processing time chunks"
        ):
            # Get cropland mask once per year if filtering enabled
            if self.cropland_filter:
                cropland_mask = cropland_masks.get(
                    year_int, np.ones_like(admin_mask, dtype=bool)
                )
                combined_mask = (admin_mask >= 0) & cropland_mask
            else:
                combined_mask = admin_mask >= 0

            # Extract all week indices for this year
            week_indices = [info[0] for info in week_info_list]
            weeks = [info[1] for info in week_info_list]

            # Get all data for this year at once - shape: (num_weeks, lat, lon)
            year_data = var_data.isel(week=week_indices).values

            # Pre-compute time strings for all weeks
            jan1 = datetime.date(year_int, 1, 1)
            time_strings = []
            for week_num in weeks:
                week_start = jan1 + datetime.timedelta(weeks=week_num - 1)
                time_strings.append(week_start.strftime("%Y-%m-%d"))

            # Process all weeks for this year in batch
            year_results = self._compute_vectorized_averages_batch(
                year_data, admin_mask, combined_mask, year_int, time_strings
            )
            results.extend(year_results)

        return results

    def _extract_all_years_from_dataset(self, dataset: xr.Dataset) -> List[int]:
        """Extract all unique years from weekly aggregated dataset"""
        if "year" in dataset.dims:
            # Weekly aggregated data with year dimension
            years = set(dataset.year.values)
        elif "year" in dataset.coords:
            # Weekly aggregated data with year coordinate
            years = set(np.unique(dataset.year.values))
        else:
            raise ValueError(
                "Dataset must have 'year' dimension or coordinate for weekly aggregated data"
            )

        return sorted(list(years))

    def _precompute_admin_arrays(self):
        """Pre-compute admin info as numpy arrays for fast access (optimization #1)"""
        n_admin = len(self.boundaries_indexed)
        
        # Pre-allocate arrays
        self.admin_names = np.empty(n_admin, dtype=object)
        self.admin_lats = np.empty(n_admin, dtype=np.float32)
        self.admin_lons = np.empty(n_admin, dtype=np.float32)
        
        # Pre-compute admin level names arrays
        self.admin_level_names = {}
        for level in range(1, self.base_processor.admin_level + 1):
            self.admin_level_names[level] = np.empty(n_admin, dtype=object)
        
        # Fill arrays once
        for admin_id in range(n_admin):
            admin_row = self.boundaries_indexed.iloc[admin_id]
            centroid = self.boundary_centroids.iloc[admin_id]
            
            self.admin_names[admin_id] = self.base_processor.get_admin_name(admin_row)
            self.admin_lats[admin_id] = round(centroid.y, 3)
            self.admin_lons[admin_id] = round(centroid.x, 3)
            
            for level in range(1, self.base_processor.admin_level + 1):
                self.admin_level_names[level][admin_id] = admin_row.get(f"NAME_{level}", "")

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
        """Create admin boundary mask for the grid with configurable subcell area weighting"""
        # Create subdivision for each grid cell (SUBCELL_SIZE^2 subcells total)
        subcell_factor = SUBCELL_SIZE
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
                    total_subcells = SUBCELL_SIZE * SUBCELL_SIZE
                    area_weights[i, j, unique_ids] = counts / total_subcells

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

    def _compute_vectorized_averages_batch(
        self,
        year_data: np.ndarray,
        admin_mask: np.ndarray,
        combined_mask: np.ndarray,
        year: int,
        time_strings: List[str],
    ) -> List[dict]:
        """Compute area-weighted averages with optimized week processing (optimized #1 & #2)"""
        num_weeks = year_data.shape[0]

        # Pre-extract area weights for all admin units (avoid repeated indexing)
        all_possible_admin_ids = np.unique(admin_mask[admin_mask >= 0])
        preloaded_weights = self.area_weights[:, :, all_possible_admin_ids]

        # Pre-allocate result arrays (optimization #2)
        max_results = num_weeks * len(all_possible_admin_ids)  # Upper bound
        result_times = np.empty(max_results, dtype=object)
        result_admin_ids = np.empty(max_results, dtype=np.int32)
        result_values = np.empty(max_results, dtype=np.float32)
        result_count = 0

        # Process each week with pre-extracted weights
        for week_idx, time_str in enumerate(time_strings):
            data_slice = year_data[week_idx]

            # Skip if all NaN
            if np.isnan(data_slice).all():
                continue

            # Get unique admin IDs that have valid data for this week
            valid_admin_ids = np.unique(
                admin_mask[combined_mask & ~np.isnan(data_slice)]
            )
            valid_admin_ids = valid_admin_ids[valid_admin_ids >= 0]

            if len(valid_admin_ids) == 0:
                continue

            # Create master mask for valid data for this week
            master_valid_mask = combined_mask & ~np.isnan(data_slice)

            # Map valid admin IDs to preloaded indices
            valid_indices_in_preloaded = np.searchsorted(
                all_possible_admin_ids, valid_admin_ids
            )
            relevant_weights = preloaded_weights[:, :, valid_indices_in_preloaded]

            # Memory-efficient computation avoiding ALL large 3D intermediate arrays
            masked_data = np.where(master_valid_mask, data_slice, 0)

            # Use einsum with mask to avoid creating masked_weights entirely
            # This computes weighted_sums and total_weights without any 3D arrays
            weighted_sums = np.einsum(
                "ij,ijk,ij->k", masked_data, relevant_weights, master_valid_mask
            )
            total_weights = np.einsum(
                "ij,ijk->k", master_valid_mask.astype(np.float32), relevant_weights
            )

            # Compute averages where weights > 0
            valid_weight_mask = total_weights > 0
            avg_values = np.divide(
                weighted_sums,
                total_weights,
                out=np.zeros_like(weighted_sums),
                where=valid_weight_mask,
            )

            # Store results in pre-allocated arrays
            valid_indices = np.where(valid_weight_mask)[0]
            n_valid = len(valid_indices)
            
            # Batch fill arrays
            end_idx = result_count + n_valid
            result_times[result_count:end_idx] = time_str
            result_admin_ids[result_count:end_idx] = valid_admin_ids[valid_indices]
            result_values[result_count:end_idx] = avg_values[valid_indices]
            result_count = end_idx

        # Convert pre-allocated arrays to list of dicts (optimization #1: use cached arrays)
        results = []
        for i in range(result_count):
            admin_id = int(result_admin_ids[i])
            
            result_dict = {
                "time": result_times[i],
                "admin_name": self.admin_names[admin_id],
                "admin_id": admin_id,
                "value": float(result_values[i]),
                "country_name": self.base_processor.country_full_name,
                "latitude": float(self.admin_lats[admin_id]),
                "longitude": float(self.admin_lons[admin_id]),
            }

            # Add admin level names using cached arrays
            for level in range(1, self.base_processor.admin_level + 1):
                result_dict[f"admin_level_{level}_name"] = self.admin_level_names[level][admin_id]

            results.append(result_dict)

        return results

    def aggregate_file(self, file_path: str, variable_name: str) -> pd.DataFrame:
        """Aggregate a single NetCDF file to administrative boundaries"""
        logger.debug(f"Aggregating file: {file_path}")

        with xr.open_dataset(file_path) as dataset:
            return self.aggregate_dataset(dataset, variable_name)
