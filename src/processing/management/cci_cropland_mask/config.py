"""CCI cropland mask processing configuration"""

from pathlib import Path
from typing import List, Optional

from src.processing.base.config import TimeSeriesConfig


class CCICroplandMaskConfig(TimeSeriesConfig):
    """Configuration for CCI cropland mask data processing"""

    # Crop classes from CCI Land Cover
    CROPLAND_CLASSES = [10, 11, 12, 20, 30]  # All cropland types
    IRRIGATED_CLASS = 20  # Irrigated cropland only

    def __init__(
        self,
        country: str,
        start_year: int,
        end_year: int,
        admin_level: int,
        data_dir: Optional[Path],
        output_format: str,
        debug: bool,
    ):
        super().__init__(
            country,
            start_year,
            end_year,
            None,  # No variables for crop mask
            admin_level,
            data_dir,
            output_format,
            debug,
        )

    def validate(self) -> None:
        """Validate crop mask-specific configuration"""
        super().validate()

    def get_cci_cropland_mask_directory(self) -> Path:
        """Get CCI cropland mask data directory for this country"""
        from src.utils.geography import Geography

        geography = Geography()
        country_full_name = geography.get_country_full_name(self.country).lower()

        # Try country-specific raw directory first
        country_cci_cropland_mask_dir = (
            self.data_dir / country_full_name / "raw" / "cci_cropland_mask"
        )
        if country_cci_cropland_mask_dir.exists():
            return country_cci_cropland_mask_dir

        raise ValueError(
            f"No CCI cropland mask data directory found for {self.country}"
        )
