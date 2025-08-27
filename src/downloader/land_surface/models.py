"""Land surface data models and enums"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class LandSurfaceVariable(Enum):
    """Available land surface variables from ERA5 single levels daily statistics"""

    LAI_HIGH_VEG = ("leaf_area_index_high_vegetation", "daily_mean")
    LAI_LOW_VEG = ("leaf_area_index_low_vegetation", "daily_mean")

    def __init__(self, variable: str, statistic: Optional[str] = None):
        self.variable = variable
        self.statistic = statistic

    @property
    def key(self):
        return self.name.lower()  # Use enum name as key


@dataclass
class DownloadConfig:
    """Download configuration for ERA5 single levels daily statistics"""

    # dataset_name: str = "derived-era5-single-levels-daily-statistics"
    dataset_name: str = "derived-era5-land-daily-statistics"
    # product_type: str = "reanalysis"
    product_type: Optional[str] = None
    daily_statistic: str = "daily_mean"
    time_zone: str = "utc+00:00"
    frequency: str = "6_hourly"
