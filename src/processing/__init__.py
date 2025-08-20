"""Processing module for aggregating weather and soil data to administrative boundaries"""

from src.processing.base_processor import BaseProcessor
from src.processing.spatial_aggregator import SpatialAggregator
from src.processing.temporal_aggregator import TemporalAggregator
from src.processing.weather_processor import WeatherProcessor

__all__ = ["BaseProcessor", "SpatialAggregator", "TemporalAggregator", "WeatherProcessor"]