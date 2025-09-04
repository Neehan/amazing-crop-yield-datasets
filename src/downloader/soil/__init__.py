"""Soil data downloader package"""

from .downloader import SoilDownloader
from .models import (
    SoilProperty,
    SoilDepth,
    PROPERTY_DEPTH_MAPPING,
    MEAN_STATISTIC,
)

__all__ = [
    "SoilDownloader",
    "SoilProperty",
    "SoilDepth",
    "PROPERTY_DEPTH_MAPPING",
    "MEAN_STATISTIC",
]
