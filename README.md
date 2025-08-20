# Weather Data Downloader

Clean, modern tool for downloading global AgERA5 weather data.

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- CDS API account and key (see setup below)

## CDS API Setup

1. Register at https://cds.climate.copernicus.eu
2. **Accept the license agreement** for the AgERA5 dataset at: https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators
3. Get your API key from your profile page
4. Create the config file:
   ```bash
   nano ~/.cdsapirc
   ```
5. Add these lines (replace with your actual key):
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_API_KEY_HERE
   ```
6. Save and exit (Ctrl+O, Enter, Ctrl+X)


## Usage

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

## Available Variables

- `temp_min` - Minimum temperature (2m_temperature, 24_hour_minimum)
- `temp_max` - Maximum temperature (2m_temperature, 24_hour_maximum)
- `precipitation` - Precipitation flux (precipitation_flux, 24_hour_mean)  
- `snow_lwe` - Snow liquid water equivalent (snow_thickness_lwe, 24_hour_mean)
- `solar_radiation` - Solar radiation flux (solar_radiation_flux, 24_hour_mean)
- `vapor_pressure` - Vapor pressure (vapour_pressure, 24_hour_mean)

See all available variables with: `python download_weather.py --list-variables`

## Features

- **Global data**: Downloads worldwide AgERA5 data (no country restrictions)
- **Year-based downloads**: Downloads each variable year by year for transparent progress
- **Progress tracking**: Real-time progress bars showing current download
- **Skip existing**: Automatically skips already downloaded files
- **Proper logging**: Uses logging module with INFO/DEBUG levels
- **Transparent errors**: No error suppression - all CDS API errors shown directly
- **AgERA5 v2.0**: Uses latest version of the dataset

## Output

Files saved to `data/weather/`:
```
data/
└── weather/
    ├── 2020_temp_min.nc
    ├── 2020_temp_max.nc
    ├── 2020_precipitation.nc
    ├── 2021_temp_min.nc
    └── ...
```