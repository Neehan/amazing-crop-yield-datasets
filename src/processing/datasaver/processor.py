"""Data saver processor for merging weather, land surface, and soil data in chunks"""

import logging
import random
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict, Any
import pandas as pd
from tqdm import tqdm

from src.processing.datasaver.config import DataSaverConfig
from src.processing.base.processor import BaseProcessor
from src.constants import DEFAULT_CHUNK_SIZE, WEATHER_END_YEAR_MAX

logger = logging.getLogger(__name__)


class DataSaverProcessor(BaseProcessor):
    """Processor for merging all data types in memory-efficient chunks"""

    def __init__(self, config: DataSaverConfig):
        """Initialize data saver processor"""
        super().__init__(config.country, config.admin_level, config.data_dir)
        self.config = config

        # Set up logging
        log_level = logging.DEBUG if config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info(f"DataSaverProcessor initialized for {config.country}")

    def process(self) -> List[Path]:
        """Process data merging in chunks"""
        logger.info(
            f"Starting data merging for {self.config.country} "
            f"({self.config.start_year}-{self.config.end_year})"
        )

        self.config.validate()

        # Setup paths
        processed_dir = self.config.get_processed_directory()
        if not processed_dir.exists():
            raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

        # Get all locations that exist in ALL datasets
        logger.info("Discovering all locations...")
        all_locations = self._get_intersection_locations(processed_dir)

        if not all_locations:
            raise ValueError("No common locations found across all datasets")

        # Shuffle locations for random distribution
        locations_list = list(all_locations)
        random.shuffle(locations_list)
        logger.info(
            f"Will process {len(locations_list)} locations in chunks of {self.config.chunk_size}"
        )

        # Create output directory
        final_dir = self.config.get_final_directory()
        final_dir.mkdir(parents=True, exist_ok=True)

        output_files = []

        # Process in chunks
        chunk_num = 0
        for i in tqdm(
            range(0, len(locations_list), self.config.chunk_size),
            desc="Processing chunks",
        ):
            chunk_locations = locations_list[i : i + self.config.chunk_size]
            chunk_num += 1

            logger.debug(
                f"Processing chunk {chunk_num}: {len(chunk_locations)} locations"
            )

            # Load data for this chunk
            weather_data = self._load_weather_data(processed_dir, chunk_locations)
            ls_data = self._load_land_surface_data(processed_dir, chunk_locations)
            soil_data = self._load_soil_data(processed_dir, chunk_locations)

            # Merge all data types
            merged_data = self._merge_all_data(weather_data, ls_data, soil_data)

            if not merged_data.empty:
                # Save chunk
                chunk_filename = (
                    f"merged_data_chunk_{chunk_num:03d}.{self.config.output_format}"
                )
                chunk_path = final_dir / chunk_filename

                if self.config.output_format == "csv":
                    merged_data.to_csv(chunk_path, index=False)
                else:
                    merged_data.to_parquet(chunk_path, index=False)

                output_files.append(chunk_path)
                logger.debug(
                    f"Saved chunk {chunk_num}: {chunk_path} ({merged_data.shape[0]} rows)"
                )
            else:
                logger.warning(f"Chunk {chunk_num} resulted in empty data")

        logger.info(
            f"Processing complete. {len(output_files)} merged files saved in: {final_dir}"
        )
        return output_files

    def _get_intersection_locations(
        self, processed_dir: Path
    ) -> Set[Tuple[str, str, str]]:
        """Get locations that exist in ALL datasets (weather, land surface, soil)"""
        csv_dir = processed_dir / "csvs"

        # Get locations from each data type
        weather_locations = self._get_locations_from_weather_files(csv_dir)
        land_surface_locations = self._get_locations_from_land_surface_files(csv_dir)
        soil_locations = self._get_locations_from_soil_files(csv_dir)

        # Only keep locations that exist in ALL datasets
        common_locations = weather_locations & land_surface_locations & soil_locations

        logger.info(
            f"Found locations - Weather: {len(weather_locations)}, Land Surface: {len(land_surface_locations)}, Soil: {len(soil_locations)}"
        )
        logger.info(f"Common locations across all datasets: {len(common_locations)}")

        return common_locations

    def _get_locations_from_weather_files(
        self, csv_dir: Path
    ) -> Set[Tuple[str, str, str]]:
        """Get all locations from weather CSV files"""
        # Weather files: weather_YYYY-YYYY_variable_weekly_weighted_adminX.csv
        pattern = f"weather_*_weekly_weighted_admin{self.config.admin_level}.csv"
        files = self.get_csv_files_by_pattern(csv_dir, pattern)

        locations = set()
        for file_path in files:
            logger.debug(f"Reading weather locations from {file_path}")
            df = pd.read_csv(file_path)

            # Extract location columns
            if "admin_level_2" in df.columns:
                file_locations = set(
                    zip(df["country"], df["admin_level_1"], df["admin_level_2"])
                )
            elif "admin_level_1" in df.columns:
                file_locations = set(zip(df["country"], df["admin_level_1"], [""]))
            else:
                file_locations = set(zip(df["country"], [""], [""]))

            locations.update(file_locations)

        return locations

    def _get_locations_from_land_surface_files(
        self, csv_dir: Path
    ) -> Set[Tuple[str, str, str]]:
        """Get all locations from land surface CSV files"""
        # Land surface files: land_surface_YYYY-YYYY_variable_weekly_weighted_adminX.csv
        pattern = f"land_surface_*_weekly_weighted_admin{self.config.admin_level}.csv"
        files = self.get_csv_files_by_pattern(csv_dir, pattern)

        locations = set()
        for file_path in files:
            logger.debug(f"Reading land surface locations from {file_path}")
            df = pd.read_csv(file_path)

            # Extract location columns
            if "admin_level_2" in df.columns:
                file_locations = set(
                    zip(df["country"], df["admin_level_1"], df["admin_level_2"])
                )
            elif "admin_level_1" in df.columns:
                file_locations = set(zip(df["country"], df["admin_level_1"], [""]))
            else:
                file_locations = set(zip(df["country"], [""], [""]))

            locations.update(file_locations)

        return locations

    def _get_locations_from_soil_files(
        self, csv_dir: Path
    ) -> Set[Tuple[str, str, str]]:
        """Get all locations from soil CSV files"""
        # Soil files: soil_property_weighted_adminX.csv
        pattern = f"soil_*_weighted_admin{self.config.admin_level}.csv"
        files = self.get_csv_files_by_pattern(csv_dir, pattern)

        locations = set()
        for file_path in files:
            logger.debug(f"Reading soil locations from {file_path}")
            df = pd.read_csv(file_path)

            # Extract location columns
            if "admin_level_2" in df.columns:
                file_locations = set(
                    zip(df["country"], df["admin_level_1"], df["admin_level_2"])
                )
            elif "admin_level_1" in df.columns:
                file_locations = set(zip(df["country"], df["admin_level_1"], [""]))
            else:
                file_locations = set(zip(df["country"], [""], [""]))

            locations.update(file_locations)

        return locations

    def _load_weather_data(
        self, processed_dir: Path, locations: List[Tuple[str, str, str]]
    ) -> Optional[pd.DataFrame]:
        """Load weather data for specified locations"""
        csv_dir = processed_dir / "csvs"

        # Weather files: weather_YYYY-YYYY_variable_weekly_weighted_adminX.csv
        pattern = f"weather_*_weekly_weighted_admin{self.config.admin_level}.csv"
        files = self.get_csv_files_by_pattern(csv_dir, pattern)

        # Load all weather files and merge them horizontally by location-year
        weather_data = None
        for file_path in files:
            logger.debug(f"Loading weather data from {file_path}")
            df = pd.read_csv(file_path)

            # Filter to specified locations
            location_mask = self._create_location_mask(df, locations)
            filtered_df = df[location_mask].copy()

            if weather_data is None:
                weather_data = filtered_df
            else:
                # Merge on location columns
                merge_cols = [
                    "country",
                    "admin_level_1",
                    "admin_level_2",
                    "year",
                    "latitude",
                    "longitude",
                ]
                merge_cols = [
                    col
                    for col in merge_cols
                    if col in weather_data.columns and col in filtered_df.columns
                ]
                weather_data = weather_data.merge(
                    filtered_df, on=merge_cols, how="outer"
                )

        if weather_data is not None:
            logger.debug(f"Loaded weather data: {weather_data.shape}")
            return pd.DataFrame(weather_data)
        else:
            raise ValueError("No weather data loaded")

    def _load_land_surface_data(
        self, processed_dir: Path, locations: List[Tuple[str, str, str]]
    ) -> Optional[pd.DataFrame]:
        """Load land surface data for specified locations"""
        csv_dir = processed_dir / "csvs"

        # Land surface files: land_surface_YYYY-YYYY_variable_weekly_weighted_adminX.csv
        pattern = f"land_surface_*_weekly_weighted_admin{self.config.admin_level}.csv"
        files = self.get_csv_files_by_pattern(csv_dir, pattern)

        # Load all land surface files and merge them horizontally by location-year
        land_surface_data = None
        for file_path in files:
            logger.debug(f"Loading land surface data from {file_path}")
            df = pd.read_csv(file_path)

            # Filter to specified locations
            location_mask = self._create_location_mask(df, locations)
            filtered_df = df[location_mask].copy()

            if land_surface_data is None:
                land_surface_data = filtered_df
            else:
                # Merge on location columns
                merge_cols = [
                    "country",
                    "admin_level_1",
                    "admin_level_2",
                    "year",
                    "latitude",
                    "longitude",
                ]
                merge_cols = [
                    col
                    for col in merge_cols
                    if col in land_surface_data.columns and col in filtered_df.columns
                ]
                land_surface_data = land_surface_data.merge(
                    filtered_df, on=merge_cols, how="outer"
                )

        if land_surface_data is not None:
            logger.debug(f"Loaded land surface data: {land_surface_data.shape}")
            return pd.DataFrame(land_surface_data)
        else:
            raise ValueError("No land surface data loaded")

    def _load_soil_data(
        self, processed_dir: Path, locations: List[Tuple[str, str, str]]
    ) -> Optional[pd.DataFrame]:
        """Load soil data for specified locations"""
        csv_dir = processed_dir / "csvs"

        # Soil files: soil_property_weighted_adminX.csv
        pattern = f"soil_*_weighted_admin{self.config.admin_level}.csv"
        files = self.get_csv_files_by_pattern(csv_dir, pattern)

        soil_data = None
        for file_path in files:
            logger.debug(f"Loading soil data from {file_path}")
            df = pd.read_csv(file_path)

            # Filter to specified locations
            location_mask = self._create_location_mask(df, locations)
            filtered_df = df[location_mask].copy()

            if soil_data is None:
                soil_data = filtered_df
            else:
                # Merge on location columns
                merge_cols = [
                    "country",
                    "admin_level_1",
                    "admin_level_2",
                    "latitude",
                    "longitude",
                ]
                merge_cols = [
                    col
                    for col in merge_cols
                    if col in soil_data.columns and col in filtered_df.columns
                ]
                soil_data = soil_data.merge(filtered_df, on=merge_cols, how="outer")

        if soil_data is not None:
            logger.debug(f"Loaded soil data: {soil_data.shape}")
            return pd.DataFrame(soil_data)
        else:
            raise ValueError("No soil data loaded")

    def _create_location_mask(
        self, df: pd.DataFrame, locations: List[Tuple[str, str, str]]
    ) -> pd.Series:
        """Create boolean mask for filtering dataframe to specified locations"""
        location_mask = pd.Series([False] * len(df))
        for country, admin1, admin2 in locations:
            mask = df["country"] == country
            if admin1:
                mask &= df["admin_level_1"] == admin1
            if admin2:
                mask &= df["admin_level_2"] == admin2
            location_mask |= mask
        return location_mask

    def _merge_all_data(
        self,
        weather_df: Optional[pd.DataFrame],
        ls_df: Optional[pd.DataFrame],
        soil_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge weather, land surface, and soil data"""

        # Start with weather data as base
        if weather_df is not None:
            final_df = weather_df.copy()
            logger.debug(f"Starting with weather data: {final_df.shape}")
        else:
            logger.error("No weather data available for merging")
            return pd.DataFrame()

        # Merge land surface data - only on administrative location + year
        if ls_df is not None:
            # Remove duplicate lat/lon from land surface data to avoid conflicts
            ls_merge_df = ls_df.drop(columns=["latitude", "longitude"], errors="ignore")
            merge_cols = ["country", "admin_level_1", "admin_level_2", "year"]

            final_df = final_df.merge(ls_merge_df, on=merge_cols, how="inner")
            logger.debug(f"After merging land surface: {final_df.shape}")

        # Merge soil data - replicate for each year in final_df
        if soil_df is not None:
            # Remove duplicate lat/lon from soil data to avoid conflicts
            soil_merge_df = soil_df.drop(
                columns=["latitude", "longitude"], errors="ignore"
            )

            # Get unique years from final_df and replicate soil for each
            years = final_df["year"].unique()
            replicated_soil_data = []
            for year in years:
                soil_year = soil_merge_df.copy()
                soil_year["year"] = year
                replicated_soil_data.append(soil_year)
            soil_replicated = pd.concat(replicated_soil_data, ignore_index=True)

            merge_cols = ["country", "admin_level_1", "admin_level_2", "year"]
            final_df = final_df.merge(soil_replicated, on=merge_cols, how="inner")
            logger.debug(f"After merging soil: {final_df.shape}")

        # Reorder columns to match the specified format
        final_df = self._reorder_columns(final_df)
        return final_df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to match the specified format:
        country, admin1, admin2, year, latitude, longitude, soil columns, weather columns, land surface columns
        """

        # Required format: country, admin1, admin2, year, latitude (3 decimal), longitude (3 decimal), soil cols, weather cols, land surface cols
        ordered_cols = [
            "country",
            "admin_level_1",
            "admin_level_2",
            "year",
            "latitude",
            "longitude",
        ]

        # Round lat/lon to 3 decimal places
        if "latitude" in df.columns:
            df["latitude"] = df["latitude"].round(3)
        if "longitude" in df.columns:
            df["longitude"] = df["longitude"].round(3)

        # Group remaining columns by variable type
        remaining_cols = [col for col in df.columns if col not in ordered_cols]

        # Separate by data type based on column patterns
        soil_cols = []
        weather_cols = []
        land_surface_cols = []
        other_cols = []

        for col in remaining_cols:
            # Soil columns typically have depth indicators (e.g., clay_0_5cm)
            if any(
                depth in col
                for depth in [
                    "_0_5cm",
                    "_5_15cm",
                    "_15_30cm",
                    "_30_60cm",
                    "_60_100cm",
                    "_100_200cm",
                ]
            ):
                soil_cols.append(col)
            # Weather columns have week indicators (e.g., precipitation_week_1)
            elif "_week_" in col and any(
                weather_var in col
                for weather_var in [
                    "precipitation",
                    "t2m_max",
                    "t2m_min",
                    "solar_radiation",
                    "vapor_pressure",
                    "wind_speed",
                    "snow_lwe",
                    "reference_et",
                ]
            ):
                weather_cols.append(col)
            # Land surface columns have week indicators and specific variables
            elif "_week_" in col and any(
                ls_var in col for ls_var in ["ndvi", "lai_low", "lai_high"]
            ):
                land_surface_cols.append(col)
            else:
                other_cols.append(col)

        # Sort each group to maintain consistent ordering
        soil_cols.sort()
        weather_cols.sort()
        land_surface_cols.sort()
        other_cols.sort()

        # Combine all columns in the specified order: soil, weather, land surface
        ordered_cols.extend(soil_cols)
        ordered_cols.extend(weather_cols)
        ordered_cols.extend(land_surface_cols)
        ordered_cols.extend(other_cols)

        # Return dataframe with reordered columns
        result_df = df[ordered_cols].copy()
        return pd.DataFrame(result_df)
