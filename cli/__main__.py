#!/usr/bin/env python3
"""Main CLI entry point for amazing-crop-yield-datasets

This allows running CLI commands via:
    python -m cli download_weather --help
    python -m cli process_weather --help
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main CLI dispatcher"""
    if len(sys.argv) < 2:
        print("Usage: python -m cli <command> [args...]")
        print("\nAvailable commands:")
        print("  download_weather  Download AgERA5 weather data")
        print(
            "  process_weather   Process weather data to county-level weekly averages"
        )
        print("\nFor help on a specific command:")
        print("  python -m cli <command> --help")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the command from sys.argv so the subcommand can parse its own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "download_weather":
        from cli.download_weather import main as download_main

        download_main()
    elif command == "process_weather":
        from cli.process_weather import main as process_main

        process_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: download_weather, process_weather")
        sys.exit(1)


if __name__ == "__main__":
    main()
