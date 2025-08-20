"""Weather data models and enums"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class WeatherVariable(Enum):
    """Available weather variables from AgERA5"""

    # Basic active variables
    T2M_MIN = ("2m_temperature", "24_hour_minimum")
    T2M_MAX = ("2m_temperature", "24_hour_maximum")
    # TEMP_MEAN = ("2m_temperature", "24_hour_mean")
    PRECIPITATION = ("precipitation_flux", "24_hour_mean")
    SNOW_LWE = ("snow_thickness_lwe", "24_hour_mean")
    SOLAR_RADIATION = ("solar_radiation_flux", "24_hour_mean")
    VAPOR_PRESSURE = ("vapour_pressure", "24_hour_mean")

    # All other AgERA5 variables (uncomment as needed)
    # TEMP_DAY_MAX = ("2m_temperature", "day_time_maximum")
    # TEMP_DAY_MEAN = ("2m_temperature", "day_time_mean")
    # TEMP_NIGHT_MEAN = ("2m_temperature", "night_time_mean")
    # TEMP_NIGHT_MIN = ("2m_temperature", "night_time_minimum")
    # DEWPOINT_TEMP = ("2m_dewpoint_temperature", "24_hour_mean")
    # CLOUD_COVER = ("cloud_cover", "24_hour_mean")
    # SNOW_THICKNESS = ("snow_thickness", "24_hour_mean")
    # WIND_SPEED = ("10m_wind_speed", "24_hour_mean")
    # REL_HUMIDITY_06H = ("2m_relative_humidity", "06_00")
    # REL_HUMIDITY_09H = ("2m_relative_humidity", "09_00")
    # REL_HUMIDITY_12H = ("2m_relative_humidity", "12_00")
    # REL_HUMIDITY_15H = ("2m_relative_humidity", "15_00")
    # REL_HUMIDITY_18H = ("2m_relative_humidity", "18_00")
    # LIQUID_PRECIP_DURATION = ("liquid_precipitation_duration_fraction", "24_hour_mean")
    # SOLID_PRECIP_DURATION = ("solid_precipitation_duration_fraction", "24_hour_mean")
    # REFERENCE_ET = ("ReferenceET_PenmanMonteith_FAO56", None)
    # REL_HUMIDITY_MIN = ("Derived_Relative_Humidity_2m_Min", None)
    # REL_HUMIDITY_MAX = ("Derived_Relative_Humidity_2m_Max", None)
    # VAPOR_PRESSURE_DEFICIT = ("Vapour_Pressure_Deficit_at_Maximum_Temperature", None)
    # PRECIP_DURATION = ("Precipitation_Duration_Fraction", None)

    def __init__(self, variable: str, statistic: str):
        self.variable = variable
        self.statistic = statistic
        self.key = self.name.lower()  # Use enum name as key


@dataclass
class DownloadConfig:
    """Download configuration"""

    dataset_name: str = "sis-agrometeorological-indicators"
    version: str = "2_0"


@dataclass
class GeoBounds:
    """Geographic bounding box"""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_cds_area(self) -> list:
        """Convert to CDS API area format [N, W, S, E]"""
        return [self.max_lat, self.min_lon, self.min_lat, self.max_lon]
