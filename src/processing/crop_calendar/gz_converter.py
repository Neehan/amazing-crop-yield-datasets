"""MIRCA2000 GZ to NetCDF converter with country filtering and monthly planting/harvesting columns"""

import logging
import gzip
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm

from src.processing.base.processor import BaseProcessor
from src.processing.crop_calendar.config import CROP_CODES
from src.utils.geography import Geography

logger = logging.getLogger(__name__)


class GZConverter:
    """Converts MIRCA2000 GZ files to NetCDF format with country filtering and monthly columns"""

    def __init__(self, base_processor: BaseProcessor):
        """Initialize converter with base processor for country boundaries

        Args:
            base_processor: Provides admin boundaries and country info
        """
        self.base_processor = base_processor
        self.country_name = base_processor.country_full_name.lower()

        # Set up directories
        self.mirca_dir = Path("data/global/mirca2000-v1.1")
        self.cache_dir = (
            Path("data") / self.country_name / "intermediate" / "crop_calendar"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get country bounds for filtering using Geography
        self.geography = Geography()
        bounds = self.geography.get_country_bounds(
            self.base_processor.country_full_name, buffer_degrees=0.5
        )
        self.bounds = {
            "min_lon": bounds.min_lon,
            "max_lon": bounds.max_lon,
            "min_lat": bounds.min_lat,
            "max_lat": bounds.max_lat,
        }

        logger.info(f"GZConverter initialized for {self.country_name}")

    def convert_crop_to_netcdf(self, crop_name: str) -> Path:
        """Convert MIRCA2000 data for a specific crop to NetCDF with weighted monthly columns

        Args:
            crop_name: Crop name (e.g., 'wheat')

        Returns:
            Path to generated NetCDF file
        """
        # Get both irrigated and rainfed crop codes for this crop
        irrigated_code = None
        rainfed_code = None

        for code, name in CROP_CODES.items():
            if name == crop_name:
                if code <= 26:  # Irrigated crops (1-26)
                    irrigated_code = code
                else:  # Rainfed crops (27-52)
                    rainfed_code = code

        if irrigated_code is None or rainfed_code is None:
            raise ValueError(
                f"Could not find both irrigated and rainfed codes for crop: {crop_name}"
            )

        # Check for cached file
        cache_filename = f"crop_calendar_{crop_name}_weighted.nc"
        cache_path = self.cache_dir / cache_filename

        if cache_path.exists():
            logger.info(f"Using cached NetCDF: {cache_path}")
            return cache_path

        logger.info(
            f"Converting crop {crop_name} (irrigated + rainfed weighted) to NetCDF"
        )

        # Load and filter data for both irrigated and rainfed
        irrigated_data = self._load_and_filter_gz_data(irrigated_code)
        rainfed_data = self._load_and_filter_gz_data(rainfed_code)

        if irrigated_data.empty and rainfed_data.empty:
            logger.warning(f"No data found for crop {crop_name} in {self.country_name}")
            # Create empty NetCDF
            self._create_empty_netcdf(cache_path, crop_name, "weighted")
            return cache_path

        # Combine and weight the data
        combined_data = self._combine_and_weight_data(irrigated_data, rainfed_data)

        # Convert to monthly planted/harvested columns
        monthly_data = self._convert_to_monthly_columns(combined_data)

        # Create NetCDF
        self._create_netcdf(monthly_data, cache_path, crop_name, "weighted")

        logger.info(f"Created weighted NetCDF: {cache_path}")
        return cache_path

    def _load_and_filter_gz_data(self, crop_code: int) -> pd.DataFrame:
        """Load GZ file and filter for country and crop code"""
        # Determine file path - try both resolutions
        gz_file_5min = self.mirca_dir / "CELL_SPECIFIC_CROPPING_CALENDARS.TXT.gz"
        gz_file_30min = self.mirca_dir / "CELL_SPECIFIC_CROPPING_CALENDARS_30MN.TXT.gz"

        # Use 5-minute resolution if available, otherwise 30-minute
        gz_file = gz_file_5min if gz_file_5min.exists() else gz_file_30min

        if not gz_file.exists():
            raise FileNotFoundError(f"MIRCA2000 data not found at {self.mirca_dir}")

        logger.info(f"Loading data from {gz_file.name}")

        # Read GZ file in chunks to handle memory
        chunk_size = 100000
        filtered_chunks = []

        with gzip.open(gz_file, "rt") as f:
            # Skip header
            f.readline()

            chunk = []
            for line in tqdm(f, desc="Processing MIRCA2000 data"):
                values = line.strip().split("\t")

                # Convert to appropriate types
                row_data = {
                    "cell_id": int(values[0]),
                    "row": int(values[1]),
                    "column": int(values[2]),
                    "lat": float(values[3]),
                    "lon": float(values[4]),
                    "crop": int(values[5]),
                    "subcrop": int(values[6]),
                    "area": float(values[7]),
                    "start": int(values[8]),
                    "end": int(values[9]),
                }

                # Filter by crop code first (most selective)
                if row_data["crop"] != crop_code:
                    continue

                # Filter by geographic bounds
                if not (
                    self.bounds["min_lat"] <= row_data["lat"] <= self.bounds["max_lat"]
                    and self.bounds["min_lon"]
                    <= row_data["lon"]
                    <= self.bounds["max_lon"]
                ):
                    continue

                chunk.append(row_data)

                # Process chunk when it gets large enough
                if len(chunk) >= chunk_size:
                    filtered_chunks.append(pd.DataFrame(chunk))
                    chunk = []

            # Add remaining data
            if chunk:
                filtered_chunks.append(pd.DataFrame(chunk))

        # Combine all chunks
        if not filtered_chunks:
            return pd.DataFrame()

        filtered_data = pd.concat(filtered_chunks, ignore_index=True)
        logger.info(f"Filtered to {len(filtered_data)} records for {self.country_name}")

        return filtered_data

    def _combine_and_weight_data(
        self, irrigated_data: pd.DataFrame, rainfed_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine irrigated and rainfed data, weighting by area to create unified crop calendar

        Args:
            irrigated_data: Irrigated crop data
            rainfed_data: Rainfed crop data

        Returns:
            Combined and weighted data
        """
        logger.info("Combining and weighting irrigated and rainfed data")

        # If one dataset is empty, return the other
        if irrigated_data.empty:
            return rainfed_data
        if rainfed_data.empty:
            return irrigated_data

        # Combine both datasets
        combined_data = pd.concat([irrigated_data, rainfed_data], ignore_index=True)

        # Group by location (lat, lon) and sum areas, weighted average start/end months
        grouped = (
            combined_data.groupby(["lat", "lon"])
            .agg(
                {
                    "area": "sum",  # Total area
                    "start": lambda x: np.average(
                        x, weights=combined_data.loc[x.index, "area"]
                    ),
                    "end": lambda x: np.average(
                        x, weights=combined_data.loc[x.index, "area"]
                    ),
                    "cell_id": "first",
                    "row": "first",
                    "column": "first",
                    "crop": "first",
                    "subcrop": "first",
                }
            )
            .reset_index()
        )

        # Round start/end months to nearest integer
        grouped["start"] = grouped["start"].round().astype(int)
        grouped["end"] = grouped["end"].round().astype(int)

        logger.info(
            f"Combined data: {len(combined_data)} records -> {len(grouped)} unique locations"
        )

        return grouped

    def _convert_to_monthly_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert start/end months to monthly planted/harvested columns as fractions"""
        logger.info("Converting to monthly planted/harvested columns")

        # Create monthly columns - initialize all to 0
        monthly_cols = {}
        for month in range(1, 13):
            monthly_cols[f"planted_month_{month}"] = np.zeros(len(data))
            monthly_cols[f"harvested_month_{month}"] = np.zeros(len(data))

        # Fill in planted and harvested months based on start/end
        for idx, row in data.iterrows():
            start_month = int(row["start"])
            end_month = int(row["end"])

            # Each crop gets a fraction of 1.0 for planting and harvesting
            # Since MIRCA2000 only gives start/end months, we assume all planting
            # happens in start month and all harvesting in end month
            monthly_cols[f"planted_month_{start_month}"][idx] = 1.0
            monthly_cols[f"harvested_month_{end_month}"][idx] = 1.0

        # Create new dataframe with original columns plus monthly columns
        result_data = data.copy()
        for col, values in monthly_cols.items():
            result_data[col] = values

        # Add dummy year column for spatial aggregation compatibility
        result_data["year"] = 2000

        return result_data

    def _create_netcdf(
        self,
        data: pd.DataFrame,
        output_path: Path,
        crop_name: str,
        irrigation_type: str,
    ):
        """Create NetCDF file from processed data"""
        # Get unique grid cells
        grid_data = (
            data.groupby(["lat", "lon"])
            .agg(
                {
                    "area": "sum",  # Total area per grid cell
                    "year": "first",
                    **{f"planted_month_{i}": "sum" for i in range(1, 13)},
                    **{f"harvested_month_{i}": "sum" for i in range(1, 13)},
                }
            )
            .reset_index()
        )

        # Get unique coordinates
        lats = sorted(grid_data["lat"].unique())
        lons = sorted(grid_data["lon"].unique())

        # Create coordinate arrays
        lat_array = np.array(lats)
        lon_array = np.array(lons)

        # Initialize data arrays
        area_data = np.zeros((len(lats), len(lons)))
        monthly_planted = np.zeros((len(lats), len(lons), 12))
        monthly_harvested = np.zeros((len(lats), len(lons), 12))

        # Fill data arrays
        for _, row in grid_data.iterrows():
            lat_idx = lats.index(row["lat"])
            lon_idx = lons.index(row["lon"])

            area_data[lat_idx, lon_idx] = row["area"]

            for month in range(1, 13):
                monthly_planted[lat_idx, lon_idx, month - 1] = row[
                    f"planted_month_{month}"
                ]
                monthly_harvested[lat_idx, lon_idx, month - 1] = row[
                    f"harvested_month_{month}"
                ]

        # Create xarray dataset
        dataset = xr.Dataset(
            {
                "area": (["lat", "lon"], area_data),
                "planted": (["lat", "lon", "month"], monthly_planted),
                "harvested": (["lat", "lon", "month"], monthly_harvested),
                "year": (["lat", "lon"], np.full((len(lats), len(lons)), 2000)),
            },
            coords={"lat": lat_array, "lon": lon_array, "month": np.arange(1, 13)},
        )

        # Add attributes
        dataset.attrs.update(
            {
                "title": f"MIRCA2000 crop calendar data for {crop_name} (weighted irrigated + rainfed)",
                "source": "MIRCA2000 v1.1",
                "crop_name": crop_name,
                "irrigation_type": irrigation_type,
                "country": self.base_processor.country_full_name,
                "reference_year": 2000,
            }
        )

        # Save to NetCDF
        dataset.to_netcdf(output_path)

    def _create_empty_netcdf(
        self, output_path: Path, crop_name: str, irrigation_type: str
    ):
        """Create empty NetCDF file when no data is found"""
        # Create minimal empty dataset
        dataset = xr.Dataset(
            {
                "area": (["lat", "lon"], np.array([[0.0]])),
                "planted": (["lat", "lon", "month"], np.zeros((1, 1, 12))),
                "harvested": (["lat", "lon", "month"], np.zeros((1, 1, 12))),
                "year": (["lat", "lon"], np.array([[2000]])),
            },
            coords={
                "lat": np.array([0.0]),
                "lon": np.array([0.0]),
                "month": np.arange(1, 13),
            },
        )

        dataset.attrs.update(
            {
                "title": f"MIRCA2000 crop calendar data for {crop_name} (weighted irrigated + rainfed) - NO DATA",
                "source": "MIRCA2000 v1.1",
                "crop_name": crop_name,
                "irrigation_type": irrigation_type,
                "country": self.base_processor.country_full_name,
                "reference_year": 2000,
            }
        )

        dataset.to_netcdf(output_path)
