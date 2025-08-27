# Amazing Crop Yield Datasets (ACYD)

Download and process weather data to county-level weekly averages for crop yield modeling.

## Setup

```bash
pip install -r requirements.txt
```

Get CDS API key:
1. Register at https://cds.climate.copernicus.eu
2. Accept AgERA5 license: https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators
3. Add key to `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_API_KEY_HERE
```

## Usage

### CLI Commands

The project provides a unified CLI interface. You can run commands in two ways:

**Option 1: Unified CLI (Recommended)**
```bash
python -m cli <command> [options]
```

**Option 2: Direct execution**
```bash
python cli/<script>.py [options]
```

### 1. Download Weather Data

```bash
# Download for a country (unified CLI)
python -m cli download_weather --country USA --start-year 2020 --end-year 2022

# Other countries
python -m cli download_weather --country Argentina --start-year 2020 --end-year 2022
python -m cli download_weather --country Brazil --start-year 2020 --end-year 2022

# Or using direct execution
python cli/download_weather.py --country USA --start-year 2020 --end-year 2022
```

**Available options:**
- `--country`: Country name (e.g., 'USA', 'Brazil', 'Argentina')
- `--start-year`: Start year (default: 1979)
- `--end-year`: End year (default: current year)
- `--variables`: Specific variables to download (default: all)
- `--concurrent`: Number of concurrent downloads (default: 4)
- `--list-variables`: List available weather variables
- `--debug`: Enable debug logging

### 2. Process to Weekly County Data

```bash
# Process to county-level weekly averages (unified CLI)
python -m cli process_weather --country USA --start-year 2020 --end-year 2022

# Process to state-level instead
python -m cli process_weather --country USA --admin-level 1 --start-year 2020 --end-year 2022

# Process specific variables only
python -m cli process_weather --country USA --variables temp_min temp_max --start-year 2020 --end-year 2022

# Or using direct execution
python cli/process_weather.py --country USA --start-year 2020 --end-year 2022
```

**Available options:**
- `--country`: Country to process (required)
- `--start-year`: Start year (default: 1979)
- `--end-year`: End year (default: current year)
- `--admin-level`: Administrative level (1=state/province, 2=county/department, default: 2)
- `--variables`: Specific variables to process (default: all)
- `--debug`: Enable debug logging

### 3. Get Help

```bash
# General help
python -m cli

# Command-specific help
python -m cli download_weather --help
python -m cli process_weather --help

# List available weather variables
python -m cli download_weather --list-variables
```

## Project Structure

```
amazing-crop-yield-datasets/
├── cli/                     # Command-line interfaces
│   ├── __init__.py         # CLI package initialization
│   ├── __main__.py         # Unified CLI entry point
│   ├── download_weather.py # Weather data download script
│   └── process_weather.py  # Weather data processing script
├── src/                    # Source code modules
│   ├── weather/           # Weather data handling
│   ├── processing/        # Data processing utilities
│   └── utils/            # Utility functions
├── data/                  # Data storage (created during processing)
└── requirements.txt       # Python dependencies
```

## Output

Creates CSV files like:
```
data/usa/processed/weather_2020-2021_temp_min_weekly_weighted_admin2.csv
data/usa/processed/weather_2020-2021_temp_max_weekly_weighted_admin2.csv
data/usa/processed/weather_2020-2021_precip_weekly_weighted_admin2.csv
```

Each file contains columns:
```
country,admin_level_1,admin_level_2,year,week_1,week_2,...,week_52
```

Ready for machine learning with 52 weekly columns per weather variable.

## Future Extensions

The CLI structure is designed to easily accommodate additional data types:
- `cli/download_soil.py` - Download soil data
- `cli/process_soil.py` - Process soil data
- `cli/download_crop.py` - Download crop yield data
- etc.
