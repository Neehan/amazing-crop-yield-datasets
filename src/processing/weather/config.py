"""Weather-specific configuration"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.processing.base.config import ProcessingConfig
from src.constants import WEATHER_START_YEAR_MIN


class WeatherConfig(ProcessingConfig):
    """Configuration for weather data processing"""

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
        """Validate weather-specific configuration"""
        super().validate()

        if self.start_year < WEATHER_START_YEAR_MIN:
            raise ValueError(
                f"start_year cannot be before {WEATHER_START_YEAR_MIN} (AgERA5 data availability)"
            )

        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")

        current_year = datetime.now().year
        if self.end_year > current_year:
            raise ValueError(
                f"end_year cannot be in the future (current year: {current_year})"
            )

    def get_weather_directory(self) -> Path:
        """Get weather data directory for this country"""
        from src.utils.geography import Geography
        geography = Geography()
        country_full_name = geography.get_country_full_name(self.country).lower()
        
        # Try country-specific directory first
        country_weather_dir = self.data_dir / country_full_name / "weather"
        if country_weather_dir.exists():
            return country_weather_dir

        # Fall back to global weather directory
        global_weather_dir = self.data_dir / "weather"
        if global_weather_dir.exists():
            return global_weather_dir

        raise ValueError(f"No weather data directory found for {self.country}")
