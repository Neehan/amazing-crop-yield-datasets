# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a global weather data downloader for agricultural datasets, specifically designed to download AgERA5 climate data from the Copernicus Climate Data Store (CDS) API. The project provides clean APIs for downloading worldwide weather data with year-based iteration and proper logging.

## Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure CDS API (required for downloads)
# 1. Register at https://cds.climate.copernicus.eu
# 2. Get API key from profile page  
# 3. Create ~/.cdsapirc with:
#    url: https://cds.climate.copernicus.eu/api/v2
#    key: YOUR_API_KEY_HERE
```

## Common Commands

```bash
# Download all weather variables (2020-2021)
python download_weather.py --start-year 2020 --end-year 2021

# Download specific variables  
python download_weather.py --variables temp_min temp_max precipitation

# Download full dataset (1979-present)
python download_weather.py

# List all available weather variables
python download_weather.py --list-variables

# Enable debug logging (shows CDS API requests)
python download_weather.py --start-year 2020 --end-year 2021 --debug
```

## Code Architecture

### Core Components

- **`src/weather/`** - Main weather module
  - `__init__.py` - Clean API with `download_weather()` function
  - `downloader.py` - `WeatherDownloader` class handling CDS API interactions
  - `models.py` - Data models (`WeatherVariable` enum, `DownloadConfig`)  
  - `geography.py` - `Geography` class (legacy - not used for global downloads)

- **`download_weather.py`** - CLI entry point with argparse interface and logging setup

### Key Design Patterns

**Year-based Downloads**: Downloads each variable for each year individually, providing transparent progress and enabling smart caching of existing files.

**Global Data Access**: Downloads worldwide AgERA5 data without geographic restrictions, using the official CDS API format.

**Proper Logging**: Uses Python logging module with configurable levels (INFO/DEBUG) instead of print statements.

**Transparent Error Handling**: No defensive error suppression - all CDS API errors are shown directly to users.

**Smart Caching**: Automatically skips existing files to avoid re-downloading data.

### Data Flow

1. **Variable Selection**: User selects variables from AgERA5 enum or defaults to all available
2. **Year Iteration**: For each variable, iterate through each year in the specified range
3. **CDS Request Building**: Create proper AgERA5 v2.0 request with correct statistic parameters
4. **File Caching**: Check if file already exists and skip if found
5. **Global Download**: Download worldwide data for the specific variable and year
6. **File Organization**: Saves to `data/weather/` with naming pattern `{year}_{variable_key}.nc`

## Weather Variables

Available variables from AgERA5 v2.0:
- `temp_min` - Minimum temperature (2m_temperature, 24_hour_minimum)
- `temp_max` - Maximum temperature (2m_temperature, 24_hour_maximum)  
- `precipitation` - Precipitation flux (precipitation_flux, 24_hour_mean)
- `snow_lwe` - Snow liquid water equivalent (snow_thickness_lwe, 24_hour_mean)
- `solar_radiation` - Solar radiation flux (solar_radiation_flux, 24_hour_mean)
- `vapor_pressure` - Vapor pressure (vapour_pressure, 24_hour_mean)

Additional variables available (uncommented in models.py as needed)

## File Structure

```
data/
└── weather/           # Global weather data
    ├── {year}_{variable_key}.nc
    ├── 2020_temp_min.nc
    ├── 2020_temp_max.nc
    └── 2021_precipitation.nc
src/
├── weather/           # Main weather downloading module
├── crop_yield/        # Crop yield data (Argentina, USA subdirs)  
└── soil/              # Soil data module
```

## Development Notes

- Uses AgERA5 v2.0 dataset from Copernicus Climate Data Store
- NetCDF files are the primary output format for weather data
- CDS API credentials are required - downloads will fail without proper ~/.cdsapirc setup
- Global downloads can be large - plan storage accordingly
- Year range is [start_year, end_year) - excludes end_year
- Uses proper logging module - enable debug with --debug flag
- All CDS API requests match the official format from the web interface