"""Weather data processor for aggregating to administrative boundaries"""

import logging
from pathlib import Path
from typing import Optional, List, Literal
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.processing.base_processor import BaseProcessor
from src.processing.spatial_aggregator import SpatialAggregator
from src.processing.temporal_aggregator import TemporalAggregator

logger = logging.getLogger(__name__)


class WeatherProcessor(BaseProcessor):
    """Processes weather data to administrative boundaries with weekly aggregation"""
    
    def __init__(
        self,
        country: str,
        admin_level: int = 2,
        data_dir: Optional[Path] = None
    ):
        """Initialize weather processor
        
        Args:
            country: Country name or ISO code (e.g., 'USA', 'Argentina', 'ARG')
            admin_level: GADM administrative level (0=country, 1=state/province, 2=county/department/municipality)
            data_dir: Base data directory (defaults to ./data)
        """
        super().__init__(country, admin_level, data_dir)
        self.spatial_aggregator = SpatialAggregator(self)
        self.temporal_aggregator = TemporalAggregator()
        
    def process_weather_data(
        self,
        start_year: int,
        end_year: int,
        variables: Optional[List[str]] = None,
        spatial_aggregation: Literal["mean", "min", "max"] = "mean",
        output_format: Literal["csv", "parquet"] = "csv"
    ) -> Path:
        """Process weather data to weekly county-level aggregates
        
        Args:
            start_year: First year to process
            end_year: Last year to process (inclusive)
            variables: List of weather variable keys (defaults to all available)
            spatial_aggregation: How to aggregate spatially within admin boundaries
            output_format: Output file format
            
        Returns:
            Path to output file
        """
        logger.info(f"Processing weather data for {self.country_iso} ({start_year}-{end_year})")
        
        # Find available weather files
        weather_dir = self._get_weather_directory()
        available_files = self._find_weather_files(weather_dir, start_year, end_year, variables)
        
        if not available_files:
            raise ValueError(f"No weather files found in {weather_dir} for years {start_year}-{end_year}")
        
        logger.info(f"Found {len(available_files)} weather files to process")
        
        # Process each file and collect results
        all_results = []
        
        for file_path in tqdm(available_files, desc="Processing weather files"):
            try:
                result = self._process_single_file(file_path, spatial_aggregation)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        if not all_results:
            raise ValueError("No weather files could be processed successfully")
        
        # Combine all results
        logger.info("Combining results from all files...")
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Aggregate to weekly format
        logger.info("Aggregating to weekly format...")
        weekly_df = self.temporal_aggregator.daily_to_weekly_pivot(combined_df, self.admin_level)
        
        # Save output
        output_path = self._save_output(weekly_df, start_year, end_year, output_format)
        logger.info(f"Saved processed weather data to: {output_path}")
        
        return output_path
    
    def _get_weather_directory(self) -> Path:
        """Get weather data directory for this country"""
        # Try country-specific directory first
        country_weather_dir = self.data_dir / self.country_iso.lower() / "weather"
        if country_weather_dir.exists():
            return country_weather_dir
        
        # Fall back to global weather directory
        global_weather_dir = self.data_dir / "weather"
        if global_weather_dir.exists():
            return global_weather_dir
            
        raise ValueError(f"No weather data directory found for {self.country_iso}")
    
    def _find_weather_files(
        self,
        weather_dir: Path,
        start_year: int,
        end_year: int,
        variables: Optional[List[str]] = None
    ) -> List[Path]:
        """Find weather NetCDF files matching criteria"""
        files = []
        
        for year in range(start_year, end_year + 1):
            year_files = list(weather_dir.glob(f"{year}_*.nc"))
            
            if variables:
                # Filter by specified variables
                year_files = [f for f in year_files if any(var in f.name for var in variables)]
            
            files.extend(year_files)
        
        return sorted(files)
    
    def _process_single_file(self, file_path: Path, spatial_aggregation: str) -> Optional[pd.DataFrame]:
        """Process a single NetCDF weather file"""
        logger.debug(f"Processing {file_path.name}")
        
        # Extract variable and year from filename
        filename = file_path.stem  # Remove .nc extension
        parts = filename.split('_', 1)
        if len(parts) != 2:
            logger.warning(f"Unexpected filename format: {filename}")
            return None
            
        year, variable = parts
        
        try:
            # Load weather data
            ds = xr.open_dataset(file_path)
            
            # Spatially aggregate to admin boundaries
            admin_data = self.spatial_aggregator.aggregate_dataset(
                ds, spatial_aggregation
            )
            
            # admin_data is already a DataFrame from spatial aggregator
            df = admin_data.copy()
            df['country'] = self.country_iso
            df['year'] = int(year)
            df['variable'] = variable
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _save_output(
        self,
        df: pd.DataFrame,
        start_year: int,
        end_year: int,
        output_format: str
    ) -> Path:
        """Save processed data to file"""
        output_dir = self.data_dir / "processed" / self.country_iso.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"weather_weekly_{start_year}_{end_year}.{output_format}"
        output_path = output_dir / filename
        
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_path