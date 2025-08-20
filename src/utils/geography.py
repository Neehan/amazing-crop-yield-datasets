"""Geographic utilities for country boundaries"""

import geopandas as gpd
from dataclasses import dataclass
from typing import List


@dataclass
class GeoBounds:
    """Geographic bounding box coordinates"""

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def to_cds_area(self) -> List[float]:
        """Convert to CDS API area format [max_lat, min_lon, min_lat, max_lon]"""
        return [self.max_lat, self.min_lon, self.min_lat, self.max_lon]


class Geography:
    """Handles country geographic data and coordinate lookups"""

    def __init__(self):
        """Initialize geography handler with lazy-loaded world data"""
        self._world_data = None

    @property
    def world_data(self):
        """Lazy load world geographic data from Natural Earth dataset"""
        if self._world_data is None:
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            self._world_data = gpd.read_file(url)
        return self._world_data

    def get_country_bounds(
        self, country_name: str, buffer_degrees: float = 0.25
    ) -> GeoBounds:
        """Get geographic bounding box for any country with optional buffer"""
        country_data = self.find_country_in_database(country_name)
        bounds = country_data.geometry.bounds

        return GeoBounds(
            min_lon=max(-180, bounds[0] - buffer_degrees),
            min_lat=max(-90, bounds[1] - buffer_degrees),
            max_lon=min(180, bounds[2] + buffer_degrees),
            max_lat=min(90, bounds[3] + buffer_degrees),
        )

    def get_country_iso_code(self, country_name: str) -> str:
        """Get ISO3 country code for a given country name"""
        country_data = self.find_country_in_database(country_name)
        return country_data["ISO_A3"]

    def get_country_full_name(self, country_identifier: str) -> str:
        """Get full country name from ISO code or country name"""
        country_data = self.find_country_in_database(country_identifier)
        return country_data["NAME"]

    def find_country_in_database(self, country_name: str):
        """Find country data"""
        world = self.world_data

        # Try exact ISO match first
        matches = world[world["ISO_A3"] == country_name.upper()]
        if not matches.empty:
            return matches.iloc[0]

        # Try exact name match
        matches = world[world["NAME"].str.upper() == country_name.upper()]
        if not matches.empty:
            return matches.iloc[0]

        # Try partial name match
        matches = world[
            world["NAME"].str.upper().str.contains(country_name.upper(), na=False)
        ]
        if not matches.empty:
            return matches.iloc[0]

        raise ValueError(f"Country '{country_name}' not found")
