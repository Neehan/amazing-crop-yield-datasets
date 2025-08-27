# Amazing Crop Yield Datasets (ACYD)

Download and process weather and land surface data to administrative-level weekly averages for crop yield modeling.

## Setup

```bash
pip install -r requirements.txt
```

### Weather Data Setup (CDS API)

Get CDS API key:
1. Register at https://cds.climate.copernicus.eu
2. Accept AgERA5 license: https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators
3. Add key to `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_API_KEY_HERE
```

### Land Surface Data Setup (Google Earth Engine)

For land surface data (LAI, NDVI, etc.) via Google Earth Engine:

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Note your PROJECT_ID

2. **Enable Earth Engine API**:
   - Go to [Google Earth Engine](https://earthengine.google.com/)
   - Sign up/login with your Google account
   - Register your cloud project for Earth Engine

3. **Authenticate**:
   ```bash
   # Authenticate (opens browser)
   earthengine authenticate
   
   # Set your project ID
   export GOOGLE_CLOUD_PROJECT=your-project-id
   ```

4. **Test authentication**:
   ```bash
   python -c "import ee; ee.Initialize(); print('✅ Earth Engine initialized successfully!')"
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

### 2. Download Land Surface Data

```bash
# Download LAI data for a country (requires Google Earth Engine authentication)
python -m cli download_land_surface --country Argentina --start-year 2020 --end-year 2022

# Download specific variables
python -m cli download_land_surface --country USA --variables lai_low lai_high --start-year 2020 --end-year 2022

# Or using direct execution
python cli/download_land_surface.py --country Argentina --start-year 2020 --end-year 2022
```

**Available options:**
- `--country`: Country name (e.g., 'USA', 'Brazil', 'Argentina')
- `--start-year`: Start year (default: 1979)
- `--end-year`: End year (default: current year)
- `--variables`: Specific variables to download (default: all available)
- `--concurrent`: Number of concurrent downloads (default: 4)
- `--list-variables`: List available land surface variables
- `--debug`: Enable debug logging

### 3. Process Weather Data

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

### 4. Process Land Surface Data

```bash
# Process LAI data to admin-level weekly averages
python -m cli process_land_surface argentina --start-year 2020 --end-year 2021

# Process specific variables only
python -m cli process_land_surface argentina --start-year 2020 --end-year 2021 --variables lai_low lai_high

# Process to state-level instead of county-level
python -m cli process_land_surface argentina --admin-level 1 --start-year 2020 --end-year 2021

# Or using direct execution
python cli/process_land_surface.py argentina --start-year 2020 --end-year 2021
```

**Available options:**
- `country`: Country to process (required, positional argument)
- `--start-year`: Start year (required)
- `--end-year`: End year (required)
- `--admin-level`: Administrative level (0=country, 1=state/province, 2=county/department, default: 1)
- `--variables`: Specific variables to process (e.g., lai_low lai_high, default: all available)
- `--output-format`: Output format (csv or parquet, default: csv)
- `--debug`: Enable debug logging

### 5. Get Help

```bash
# General help
python -m cli

# Command-specific help
python -m cli download_weather --help
python -m cli download_land_surface --help
python -m cli process_weather --help
python -m cli process_land_surface --help

# List available variables
python -m cli download_weather --list-variables
python -m cli download_land_surface --list-variables
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

## Processing Pipeline

The processing module (`src/processing/`) transforms raw gridded weather data into county-level weekly averages suitable for crop yield modeling. Here's how it works:

### Data Flow Overview

```
Raw Daily Weather Data (NetCDF)
    ↓ [Zip Extraction]
Annual NetCDF Files
    ↓ [Temporal Aggregation]
Weekly NetCDF Files
    ↓ [Spatial Aggregation]
County-Level Weekly CSV Files
```

### 1. Zip Extraction (`ZipExtractor`)

**Purpose**: Extract and organize daily weather data from zip archives
- **Input**: Zip files containing daily NetCDF files
- **Output**: Annual NetCDF files with daily data organized by year
- **Optimization**: Caches extracted files to avoid repeated extraction

### 2. Temporal Aggregation (`TemporalAggregator`)

**Purpose**: Convert daily data to weekly averages
- **Input**: Daily NetCDF files (365+ days per year)
- **Output**: Weekly NetCDF files (52 weeks per year)
- **Method**: Groups daily data by ISO week number and computes mean values
- **Memory Efficient**: Processes data in chunks to handle large datasets

### 3. Spatial Aggregation (`SpatialAggregator`)

**Purpose**: Convert gridded data to administrative boundary averages

#### Two-Stage Process:

**Stage 1: FILTERING** - Create masks to identify which grid cells to process
- **Admin Boundary Mask**: Maps each grid cell to its administrative unit (cached globally)
- **Cropland Mask**: Identifies grid cells with significant cropland (cached per year using HYDE data)
- **Combined Mask**: Only processes admin units that contain cropland

**Stage 2: AVERAGING** - Compute area-weighted averages for each administrative unit
- **25-Subcell Subdivision**: Each grid cell is divided into 5×5 subcells for accurate area weighting
- **Vectorized Computation**: Processes all administrative units simultaneously using numpy operations
- **Memory Optimized**: Caches admin info to avoid repeated pandas lookups

#### Key Optimizations:

1. **Mask Caching**: Admin boundary and cropland masks are cached as NetCDF files
2. **Area Weighting**: Uses 25-subcell subdivision for precise area calculations
3. **Cropland Filtering**: Only processes areas with >1% cropland coverage
4. **Vectorized Operations**: Batch processing of all admin units per time step
5. **Admin Info Caching**: Pre-computes admin metadata to avoid expensive pandas operations

#### Technical Details:

- **Grid Resolution**: Typically 0.1° × 0.1° (AgERA5 resolution)
- **Administrative Levels**: 
  - Level 1: States/Provinces
  - Level 2: Counties/Departments (default)
- **Data Sources**:
  - **Boundaries**: GADM (Global Administrative Areas) database
  - **Cropland**: HYDE 3.5 historical land use dataset
- **Coordinate System**: WGS84 (EPSG:4326)

### 4. Formatting (`BaseFormatter`)

**Purpose**: Transform aggregated data into final CSV format
- **Input**: Time series data with admin boundaries and values
- **Output**: Pivot tables with admin hierarchy and weekly columns
- **Format**: `country,admin_level_1,admin_level_2,lat,lon,year,week_1,week_2,...,week_52`

### Performance Characteristics

- **Memory Usage**: Processes data in chunks to handle large countries
- **Caching Strategy**: Extensive caching of masks and intermediate results
- **Scalability**: Handles countries from small (e.g., Uruguay) to large (e.g., USA, Brazil)
- **Processing Time**: 
  - Small countries: ~5-10 minutes per variable per year
  - Large countries: ~30-60 minutes per variable per year

### Weather Variables Supported

The system processes various AgERA5 meteorological variables:
- **Temperature**: `temp_min`, `temp_max`, `temp_mean`
- **Precipitation**: `precip`
- **Solar Radiation**: `solar_radiation`
- **Wind**: `wind_speed`
- **Humidity**: `dewpoint_temp`, `vapour_pressure`
- **And more**: See `--list-variables` for complete list

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
