"""Configuration for data saver (chunked merging)"""

from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    DEFAULT_ADMIN_LEVEL,
    DEFAULT_START_YEAR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_CHUNK_SIZE,
    WEATHER_END_YEAR_MAX,
)


@dataclass
class DataSaverConfig:
    """Configuration for chunked data merging"""

    country: str
    start_year: int = DEFAULT_START_YEAR
    end_year: int = WEATHER_END_YEAR_MAX
    admin_level: int = DEFAULT_ADMIN_LEVEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    data_dir: Path = Path("./data")
    output_format: str = DEFAULT_OUTPUT_FORMAT
    debug: bool = False

    def __post_init__(self):
        """Ensure data_dir is a Path object"""
        if self.data_dir is None:
            self.data_dir = Path("./data")
        else:
            self.data_dir = Path(self.data_dir)

    def validate(self):
        """Validate configuration parameters"""
        if self.start_year >= self.end_year:
            raise ValueError(
                f"Start year ({self.start_year}) must be less than end year ({self.end_year})"
            )

        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")

        if self.admin_level not in [0, 1, 2]:
            raise ValueError(f"Admin level must be 0, 1, or 2, got {self.admin_level}")

    def get_processed_directory(self) -> Path:
        """Get the processed data directory for this country"""
        country_name = self.country.lower().replace(" ", "_")
        return self.data_dir / country_name / "processed"

    def get_final_directory(self) -> Path:
        """Get the final output directory"""
        return self.get_processed_directory() / "final"
