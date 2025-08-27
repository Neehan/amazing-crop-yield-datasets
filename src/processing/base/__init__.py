"""Base processing classes for weather and soil data aggregation"""

from src.processing.base.config import ProcessingConfig
from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.base.temporal_aggregator import TemporalAggregator
from src.processing.base.formatter import BaseFormatter
from src.processing.base.zip_extractor import ZipExtractor

__all__ = [
    "ProcessingConfig",
    "BaseProcessor",
    "SpatialAggregator",
    "TemporalAggregator",
    "BaseFormatter",
    "ZipExtractor",
]
