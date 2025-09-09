#!/usr/bin/env python3
"""CLI for chunked data merging (datasaver) processor"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.base_process_cli import run_processor_cli
from src.processing.datasaver.config import DataSaverConfig
from src.processing.datasaver.processor import DataSaverProcessor
from src.constants import DEFAULT_CHUNK_SIZE

import random
import numpy as np

random.seed(42)
np.random.seed(42)


def add_custom_args(parser):
    """Add datasaver-specific arguments"""
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="Start year (inclusive)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
        help="End year (exclusive)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of locations to process per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )


def parse_custom_args(args):
    """Parse custom arguments and return config kwargs"""
    return {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "chunk_size": args.chunk_size,
    }


if __name__ == "__main__":
    run_processor_cli(
        description="Merge weather, land surface, and soil data in memory-efficient chunks",
        config_class=DataSaverConfig,
        processor_class=DataSaverProcessor,
        add_custom_args_func=add_custom_args,
        parse_custom_args_func=parse_custom_args,
        success_message="Data merging completed successfully!",
    )
