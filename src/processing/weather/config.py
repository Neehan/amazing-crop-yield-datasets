"""Weather-specific configuration"""

from pathlib import Path
from typing import List, Optional

from src.processing.base.config import TimeSeriesConfig
from src.constants import WEATHER_START_YEAR_MIN


class WeatherConfig(TimeSeriesConfig):
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

    def validate(self) -> None:
        """Validate weather-specific configuration"""
        super().validate()

        if self.start_year < WEATHER_START_YEAR_MIN:
            raise ValueError(
                f"start_year cannot be before {WEATHER_START_YEAR_MIN} (AgERA5 data availability)"
            )

    def get_weather_directory(self) -> Path:
        """Get weather data directory for this country"""
        from src.utils.geography import Geography

        geography = Geography()
        country_full_name = geography.get_country_full_name(self.country).lower()

        # Try country-specific raw directory first
        country_weather_dir = self.data_dir / country_full_name / "raw" / "weather"
        if country_weather_dir.exists():
            return country_weather_dir

        raise ValueError(f"No weather data directory found for {self.country}")
