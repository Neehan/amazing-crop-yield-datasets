"""Soil-specific configuration"""

from pathlib import Path
from typing import List, Optional

from src.processing.base.config import ProcessingConfig
from src.downloader.soil import SoilProperty, SoilDepth
from src.utils.geography import Geography


class SoilConfig(ProcessingConfig):
    """Configuration for soil data processing"""

    def __init__(
        self,
        country: str,
        properties: List[str],
        depths: List[str],
        admin_level: int,
        data_dir: Path,
        output_format: str,
        debug: bool,
    ):
        super().__init__(country, admin_level, data_dir, output_format, debug)
        self.properties = properties
        self.depths = depths

    def validate(self) -> None:
        """Validate soil-specific configuration"""
        super().validate()

        valid_properties = [prop.key for prop in SoilProperty]
        invalid_props = [p for p in self.properties if p not in valid_properties]
        if invalid_props:
            raise ValueError(
                f"Invalid properties: {invalid_props}. Valid options: {valid_properties}"
            )

        valid_depths = [depth.key for depth in SoilDepth]
        invalid_depths = [d for d in self.depths if d not in valid_depths]
        if invalid_depths:
            raise ValueError(
                f"Invalid depths: {invalid_depths}. Valid options: {valid_depths}"
            )

    def get_soil_directory(self) -> Path:
        """Get soil data directory for this country"""
        geography = Geography()
        country_full_name = geography.get_country_full_name(self.country).lower()

        country_soil_dir = self.data_dir / country_full_name / "soil"
        if country_soil_dir.exists():
            return country_soil_dir

        raise ValueError(f"No soil data directory found for {self.country}")
