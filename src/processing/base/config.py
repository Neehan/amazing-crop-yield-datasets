"""Base configuration class for processing pipelines"""

from pathlib import Path
from typing import Optional

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
