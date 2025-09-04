"""Soil data models and enums for SoilGrids API"""

from enum import Enum


class SoilProperty(Enum):
    """Available soil properties from SoilGrids"""

    BULK_DENSITY = ("bdod", "Bulk density of the fine earth fraction")
    CEC = ("cec", "Cation Exchange Capacity of the soil")
    COARSE_FRAGMENTS = ("cfvo", "Volumetric fraction of coarse fragments")
    CLAY = ("clay", "Proportion of clay particles")
    NITROGEN = ("nitrogen", "Total nitrogen")
    ORGANIC_CARBON_DENSITY = ("ocd", "Organic carbon density")
    PH_H2O = ("phh2o", "Soil pH in H2O")
    SAND = ("sand", "Proportion of sand particles")
    SILT = ("silt", "Proportion of silt particles")
    ORGANIC_CARBON = ("soc", "Soil organic carbon content")
    ORGANIC_CARBON_STOCK = ("ocs", "Soil organic carbon stock")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @property
    def key(self):
        return self.name.lower()


class SoilDepth(Enum):
    """Available soil depth ranges from SoilGrids"""

    DEPTH_0_5 = ("0-5cm", 0, 5)
    DEPTH_5_15 = ("5-15cm", 5, 15)
    DEPTH_15_30 = ("15-30cm", 15, 30)
    DEPTH_30_60 = ("30-60cm", 30, 60)
    DEPTH_60_100 = ("60-100cm", 60, 100)
    DEPTH_100_200 = ("100-200cm", 100, 200)

    def __init__(self, range_str: str, start_cm: int, end_cm: int):
        self.range_str = range_str
        self.start_cm = start_cm
        self.end_cm = end_cm

    @property
    def key(self):
        return f"{self.start_cm}_{self.end_cm}cm"


# Property-depth mappings from the API response
PROPERTY_DEPTH_MAPPING = {
    SoilProperty.BULK_DENSITY: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.CEC: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.COARSE_FRAGMENTS: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.CLAY: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.NITROGEN: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.ORGANIC_CARBON_DENSITY: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.PH_H2O: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.SAND: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.SILT: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.ORGANIC_CARBON: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
    SoilProperty.ORGANIC_CARBON_STOCK: [
        SoilDepth.DEPTH_0_5,
        SoilDepth.DEPTH_5_15,
        SoilDepth.DEPTH_15_30,
        SoilDepth.DEPTH_30_60,
        SoilDepth.DEPTH_60_100,
        SoilDepth.DEPTH_100_200,
    ],
}

# We only download mean values
MEAN_STATISTIC = "mean"
