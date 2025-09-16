"""Irrigation fraction processing configuration"""

from pathlib import Path
from typing import Optional
from datetime import datetime


from src.processing.base.config import TimeSeriesConfig


class IrrigationConfig(TimeSeriesConfig):
    """Configuration for irrigation fraction data processing"""

    def __init__(
        self,
        country: str,
        start_year: int,
        end_year: int,
        variables: Optional[list],
        admin_level: int,
        data_dir: Path,
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
        # For irrigation, we process one year at a time, so use start_year as the target year
        self.year = start_year

    def validate(self) -> None:
        """Validate irrigation specific configuration"""
        super().validate()

        current_year = datetime.now().year

        if self.start_year < 1979:
            raise ValueError(
                f"Year {self.start_year} too early. Irrigation data meaningful from 1979 onwards"
            )
        if self.start_year > current_year:
            raise ValueError(
                f"Year {self.start_year} in the future. Current year: {current_year}"
            )

    def get_hyde_directory(self) -> Path:
        """Get HYDE data directory"""
        hyde_dir = self.data_dir / "global" / "hyde-3.5"
        if not hyde_dir.exists():
            raise FileNotFoundError(f"HYDE directory not found: {hyde_dir}")
        return hyde_dir

    def get_irrigation_file(self) -> Path:
        """Get irrigation data file path"""
        hyde_dir = self.get_hyde_directory()
        irrigation_file = hyde_dir / "total_irrigated.nc"
        if not irrigation_file.exists():
            raise FileNotFoundError(f"Irrigation file not found: {irrigation_file}")
        return irrigation_file

    def get_cropland_file(self) -> Path:
        """Get cropland data file path"""
        hyde_dir = self.get_hyde_directory()
        cropland_file = hyde_dir / "cropland.nc"
        if not cropland_file.exists():
            raise FileNotFoundError(f"Cropland file not found: {cropland_file}")
        return cropland_file

    def get_final_directory(self) -> Path:
        """Get final output directory for irrigation data"""
        return self.data_dir / self.country.lower() / "final" / "management"
