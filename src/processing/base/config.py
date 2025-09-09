"""Base configuration class for processing pipelines"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from src.constants import (
    ADMIN_LEVEL_COUNTRY,
    ADMIN_LEVEL_STATE,
    ADMIN_LEVEL_COUNTY,
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_PARQUET,
)


class ProcessingConfig:
    """Base configuration class with shared parameters"""

    def __init__(
        self,
        country: str,
        admin_level: int,
        data_dir: Optional[Path],
        output_format: str,
        debug: bool,
    ):
        self.country = country
        self.admin_level = admin_level
        self.data_dir = Path(data_dir or "data")
        self.output_format = output_format
        self.debug = debug

    def validate(self) -> None:
        """Validate base configuration parameters"""
        valid_admin_levels = [
            ADMIN_LEVEL_COUNTRY,
            ADMIN_LEVEL_STATE,
            ADMIN_LEVEL_COUNTY,
        ]
        if self.admin_level not in valid_admin_levels:
            raise ValueError(
                f"Invalid admin_level: {self.admin_level}. Must be one of {valid_admin_levels}"
            )

        valid_formats = [OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_PARQUET]
        if self.output_format not in valid_formats:
            raise ValueError(
                f"Invalid output_format: {self.output_format}. Must be one of {valid_formats}"
            )

        if not self.country:
            raise ValueError("Country must be specified")

    def get_intermediate_directory(self) -> Path:
        """Get intermediate directory for this country"""
        country_name = self.country.lower().replace(" ", "_")
        return self.data_dir / country_name / "intermediate"

    def get_processed_subdirectory(self, subdir: str) -> Path:
        """Get a subdirectory within intermediate"""
        processed_dir = self.get_intermediate_directory() / subdir
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir

    def get_final_directory(self) -> Path:
        """Get the final output directory"""
        country_name = self.country.lower().replace(" ", "_")
        return self.data_dir / country_name / "final"


class TimeSeriesConfig(ProcessingConfig):
    """Base configuration for time series data (weather, land surface)"""

    def __init__(
        self,
        country: str,
        start_year: int,
        end_year: int,
        variables: Optional[list],
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
        """Validate time series configuration"""
        super().validate()

        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")

        current_year = datetime.now().year
        if self.end_year > current_year:
            raise ValueError(
                f"end_year cannot be in the future (current year: {current_year})"
            )
