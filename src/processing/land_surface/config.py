"""Land surface processing configuration"""

from pathlib import Path
from typing import List, Optional

from src.processing.base.config import TimeSeriesConfig


class LandSurfaceConfig(TimeSeriesConfig):
    """Configuration for land surface data processing"""

    def __init__(
        self,
        country: str,
        start_year: int,
        end_year: int,
        variables: Optional[List[str]],
        admin_level: int,
        data_dir: Optional[Path],
        output_format: str,
        debug: bool,
    ):
        super().__init__(
            country,
            start_year,
            end_year,
            variables,
            admin_level,
            data_dir,
            output_format,
            debug,
        )

    def get_land_surface_directory(self) -> Path:
        """Get land surface data directory for this country"""
        from src.utils.geography import Geography

        geography = Geography()
        country_full_name = geography.get_country_full_name(self.country).lower()

        # Country-specific raw directory
        country_ls_dir = self.data_dir / country_full_name / "raw" / "land_surface"
        if country_ls_dir.exists():
            return country_ls_dir

        raise ValueError(f"No land surface data directory found for {self.country}")
