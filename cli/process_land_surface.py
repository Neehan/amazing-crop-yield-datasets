#!/usr/bin/env python3
"""CLI for processing land surface data"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_process_cli import run_processor_cli
from src.processing.land_surface.config import LandSurfaceConfig
from src.processing.land_surface.processor import LandSurfaceProcessor
from src.downloader.land_surface.models import LandSurfaceVariable


def main():
    run_processor_cli(
        description="Process land surface data (LAI, NDVI, etc.) to CSV format aggregated by administrative boundaries",
        config_class=LandSurfaceConfig,
        processor_class=LandSurfaceProcessor,
        variable_enum=LandSurfaceVariable,
        success_message="Land surface processing completed successfully!",
    )


if __name__ == "__main__":
    main()
