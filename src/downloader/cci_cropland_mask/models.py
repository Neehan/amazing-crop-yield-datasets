"""CCI cropland mask data models and configurations"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class CCICroplandMaskVariable(Enum):
    """Available CCI cropland mask variables"""

    CROPLAND_MASK = ("cropland_mask", "all")

    def __init__(self, key: str, variable: str):
        self.key = key
        self.variable = variable


@dataclass
class CCICroplandMaskConfig:
    """Configuration for CCI cropland mask downloads"""

    dataset_name: str = "satellite-land-cover"
    versions: Optional[List[str]] = None
    variable: str = "all"

    def __post_init__(self):
        if self.versions is None:
            self.versions = ["v2_0_7cds", "v2_1_1"]
