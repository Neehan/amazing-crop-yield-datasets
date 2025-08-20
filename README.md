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

### 1. Download Weather Data

```bash
# Download for a country
python download_weather.py --country USA --start-year 2020 --end-year 2022

# Other countries
python download_weather.py --country Argentina --start-year 2020 --end-year 2022
python download_weather.py --country Brazil --start-year 2020 --end-year 2022
```

### 2. Process to Weekly County Data

```bash
# Process to county-level weekly averages
python process_weather.py --country USA --start-year 2020 --end-year 2022

# Process to state-level instead
python process_weather.py --country USA --admin-level 1 --start-year 2020 --end-year 2022

# Process specific variables only
python process_weather.py --country USA --variables temp_min temp_max --start-year 2020 --end-year 2022
```

## Output

Creates CSV files like:
```
data/usa/processed/weather_2020-2021_all-vars_weekly_mean_admin2.csv
```

With columns:
```
country,admin_level_1,admin_level_2,year,temp_min_week_1,...,temp_min_week_52,temp_max_week_1,...
```

Ready for machine learning with 52 weekly columns per weather variable.
