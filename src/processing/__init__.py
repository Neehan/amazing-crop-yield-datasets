"""Processing module for aggregating weather and soil data to administrative boundaries

This module follows a clean inheritance hierarchy:
- Base classes in src/processing/base/ provide shared functionality
- Weather-specific classes in src/processing/weather/ inherit from base classes
- Future soil classes will go in src/processing/soil/
"""

# Import base classes
from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.base.config import ProcessingConfig
from src.processing.base.zip_extractor import ZipExtractor

# Import weather-specific classes
from src.processing.weather.config import WeatherConfig
from src.processing.weather.processor import WeatherProcessor
from src.processing.base.temporal_aggregator import TemporalAggregator

__all__ = [
    # Base classes
    "BaseProcessor",
    "SpatialAggregator",
    "ProcessingConfig",
    "ZipExtractor",
    # Weather-specific classes
    "WeatherConfig",
    "WeatherProcessor",
    "TemporalAggregator",
]
