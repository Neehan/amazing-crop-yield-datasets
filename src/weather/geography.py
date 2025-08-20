"""Geographic utilities for country boundaries"""

import geopandas as gpd
from src.weather.models import GeoBounds


class Geography:
    """Handles country geographic data and coordinate lookups"""

    def __init__(self):
        """Initialize geography handler with lazy-loaded world data"""
        self._world_data = None

    @property
    def world_data(self):
        """Lazy load world geographic data from Natural Earth dataset

        Returns:
            GeoDataFrame containing world country boundaries
        """
        if self._world_data is None:
            # Use Natural Earth 110m countries dataset directly
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            self._world_data = gpd.read_file(url)
        return self._world_data

    def get_country_bounds(
        self, country_name: str, buffer_degrees: float = 0.25
    ) -> GeoBounds:
        """Get geographic bounding box for any country with optional buffer

        Args:
            country_name: Name of country (e.g., 'USA', 'Germany', 'Argentina')
            buffer_degrees: Buffer to add around country bounds in degrees

        Returns:
            GeoBounds object with min/max longitude and latitude

        Raises:
            ValueError: If country cannot be found in geographic database

        Example:
            geography = Geography()
            bounds = geography.get_country_bounds("USA")
            print(f"USA bounds: {bounds.min_lon}, {bounds.min_lat}, {bounds.max_lon}, {bounds.max_lat}")
        """
        country_data = self.find_country_in_database(country_name)
        bounds = country_data.geometry.bounds

        # Apply buffer and clamp to valid world coordinates
        return GeoBounds(
            min_lon=max(-180, bounds[0] - buffer_degrees),
            min_lat=max(-90, bounds[1] - buffer_degrees),
            max_lon=min(180, bounds[2] + buffer_degrees),
            max_lat=min(90, bounds[3] + buffer_degrees),
        )

    def find_country_in_database(self, country_name: str):
        """Find country data using multiple matching strategies

        Tries exact name match, ISO3 code match, and partial name match.

        Args:
            country_name: Country name or ISO code to search for

        Returns:
            GeoDataFrame row containing country geographic data

        Raises:
            ValueError: If country cannot be found with any matching strategy
        """
        world = self.world_data

        # Strategy 1: Direct name match (try multiple name columns)
        name_columns = ["NAME", "NAME_EN", "NAME_LONG", "ADMIN"]
        for col in name_columns:
            if col in world.columns:
                match = world[world[col].str.lower() == country_name.lower()]
                if not match.empty:
                    return self._select_largest_geometry(match)

        # Strategy 2: ISO3 code match
        iso_columns = ["ISO_A3", "ADM0_A3"]
        for col in iso_columns:
            if col in world.columns:
                match = world[world[col].str.lower() == country_name.lower()]
                if not match.empty:
                    return self._select_largest_geometry(match)

        # Strategy 3: Partial name match
        for col in name_columns:
            if col in world.columns:
                match = world[
                    world[col].str.contains(country_name, case=False, na=False)
                ]
                if not match.empty:
                    return self._select_largest_geometry(match)

        # If still not found, show available columns and countries for debugging
        available_columns = list(world.columns)
        if "NAME" in world.columns:
            sample_countries = world["NAME"].head(10).tolist()
        elif "ADMIN" in world.columns:
            sample_countries = world["ADMIN"].head(10).tolist()
        else:
            sample_countries = "No name column found"

        raise ValueError(
            f"Country '{country_name}' not found in geographic database. "
            f"Available columns: {available_columns}. "
            f"Sample countries: {sample_countries}"
        )

    def _select_largest_geometry(self, matches):
        """Select the largest geometry from multiple matches
        
        For countries like USA with multiple geometries (continental + islands),
        this selects the largest one (continental USA).
        
        Args:
            matches: GeoDataFrame with potentially multiple rows
            
        Returns:
            Single row with the largest geometry by area
        """
        if len(matches) == 1:
            return matches.iloc[0]
        
        # Calculate area for each geometry and select the largest
        matches_with_area = matches.copy()
        matches_with_area['area'] = matches_with_area.geometry.area
        largest_idx = matches_with_area['area'].idxmax()
        
        return matches.loc[largest_idx]
