"""Land surface processing configuration"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.processing.base.config import ProcessingConfig


class LandSurfaceConfig(ProcessingConfig):
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
        super().__init__(country, admin_level, data_dir, output_format, debug)
        self.start_year = start_year
        self.end_year = end_year
        self.variables = variables

    def validate(self) -> None:
        """Validate land surface configuration"""
        super().validate()

        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")

        current_year = datetime.now().year
        if self.end_year > current_year:
            raise ValueError(
                f"end_year cannot be in the future (current year: {current_year})"
            )

    def get_land_surface_directory(self) -> Path:
        """Get land surface data directory for this country"""
        from src.utils.geography import Geography

        geography = Geography()
        country_full_name = geography.get_country_full_name(self.country).lower()

        # Country-specific directory
        country_ls_dir = self.data_dir / country_full_name / "land_surface"
        if country_ls_dir.exists():
            return country_ls_dir

        raise ValueError(f"No land surface data directory found for {self.country}")
