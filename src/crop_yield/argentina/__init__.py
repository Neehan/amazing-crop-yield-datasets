"""
Argentina crop yield data processing package.

This package provides tools for processing raw Argentina crop yield data
from the Ministerio de Agricultura, Ganadería y Pesca (MAGyP) into
standardized CSV format suitable for analysis.
"""

from src.crop_yield.argentina.models import CROP_NAME_MAPPING, SUPPORTED_CROPS

__all__ = [
    "CROP_NAME_MAPPING",
    "SUPPORTED_CROPS",
]
