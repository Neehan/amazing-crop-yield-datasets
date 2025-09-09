# Amazing Crop Yield Datasets (ACYD)

Download and process weather and land surface data to administrative-level weekly averages for crop yield modeling.

## Setup

```bash
pip install -r requirements.txt
```
### Cropland Mask Setup

Download the HYDE-3.5 cropland mask:
```bash
mkdir -p data/global
wget https://geo.public.data.uu.nl/vault-hyde/hyde35_c9_apr2025%5B1749214444%5D/original/gbc2025_7apr_base/NetCDF/cropland.nc -O data/global/cropland.nc
```

### Weather Data Setup (CDS API)

Weather data is downloaded directly from the AgERA5 reanalysis dataset. This dataset is public, freely accessible, and can be downloaded in reasonable time.

To get your CDS API key:
1. Register at https://cds.climate.copernicus.eu
2. Accept AgERA5 license: https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators
3. Add key to `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_API_KEY_HERE
```

### Land Surface Data Setup (Google Earth Engine)

Land surface data (LAI, NDVI, etc.) comes from multiple sources via Google Earth Engine:
- **LAI**: ERA-5 Daily Reanalysis (**not AgERA-5**)  
- **NDVI**: NOAA CDR datasets (AVHRR for 1982-2013, VIIRS for 2014+)

Downloading this data directly from original sources is inconvenient since each request takes over 40 minutes. Therefore, it should be downloaded via Google Earth Engine.

**Important Note:** Google Earth Engine API access is free only up to a limit. If you exceed the limit, your account will be charged. This is why the entire weather dataset is not downloaded through Google Earth Engine, even though it would be technically possible and faster.

To set up Google Earth Engine access:

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
   **Note:** If you are on an HPC that blocks I/O, the easiest option is to generate the credentials on your local computer with the above command, then copy them to `~/.config/earthengine/credentials` on remote.


## Usage

### 1. Download Weather Data

```bash
# Download for a country
python -m cli.download_weather --country USA --start-year 2020 --end-year 2022
# Note: Downloads data for years 2020 and 2021 (end-year is exclusive)

# Other countries
python -m cli.download_weather --country argentina --start-year 2020 --end-year 2022
```

**Available options:**
- `--country`: Country name (e.g., 'USA', 'Brazil', 'Argentina')
- `--start-year`: Start year inclusive (default: 1979)
- `--end-year`: End year exclusive (default: current year)
- `--variables`: Specific variables to download (default: all)
- `--concurrent`: Number of concurrent downloads (default: 8)
- `--list-variables`: List available weather variables
- `--debug`: Enable debug logging

### 2. Download Land Surface Data

*Requires Google Earth Engine authentication (see setup above)*

```bash
# Download LAI and NDVI data for a country
python -m cli.download_land_surface --country Argentina --start-year 2020 --end-year 2022

# Download specific variables
python -m cli.download_land_surface --country USA --variables lai_low lai_high ndvi --start-year 2020 --end-year 2022

# Download NDVI data spanning the dataset transition (1982-2013 AVHRR, 2014+ VIIRS)
python -m cli.download_land_surface --country USA --variables ndvi --start-year 2010 --end-year 2020
```

**Available options:**
- `--country`: Country name (e.g., 'USA', 'Brazil', 'Argentina')
- `--start-year`: Start year (default: 1979, but NDVI only available from 1982)
- `--end-year`: End year (default: current year)
- `--variables`: Specific variables to download (default: all available)
- `--concurrent`: Number of concurrent downloads (default: 4)
- `--list-variables`: List available land surface variables
- `--debug`: Enable debug logging

### 3. Download Soil Data
The soil data is from SoilGrids survey. The original survey has [11 documented soil variables](https://www.isric.org/explore/soilgrids/faq-soilgrids) at 6 depth levels. However, in practice, the OCS is only available for 0-30cm level and thus is discarded from analysis. The dataset was published in 2020 and there is no temporal component.

Here is how to download soil data:

```bash
# Download soil data for a country
python -m cli.download_soil --country USA

# Download specific properties
python -m cli.download_soil --country Argentina --properties bulk_density clay

# Download specific depths
python -m cli.download_soil --country Brazil --depths 0_5cm 5_15cm
```

**Available options:**
- `--country`: Country name (required)
- `--properties`: Soil properties to download (default: all available)
- `--depths`: Depth ranges to download (default: all available for each property)
- `--concurrent`: Number of concurrent downloads (default: 5)
- `--list-properties`: List available soil properties
- `--list-depths`: List available depth ranges
- `--debug`: Enable debug logging

### 4. Process Weather Data

```bash
# Process to county-level weekly averages
python -m cli.process_weather --country USA --start-year 2020 --end-year 2022

# Process to state-level instead
python -m cli.process_weather --country USA --admin-level 1 --start-year 2020 --end-year 2022

# Process specific variables only
python -m cli.process_weather --country USA --variables temp_min temp_max --start-year 2020 --end-year 2022
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
python -m cli.process_land_surface --country argentina --start-year 1979 --end-year 2025
# Process specific variables only
python -m cli.process_land_surface --country argentina --start-year 1982 --end-year 2025 --variables ndvi

# Process to state-level instead of county-level
python -m cli.process_land_surface --country argentina --start-year 1982 --end-year 2025 --admin-level 1
```

**Available options:**
- `country`: Country to process (required, positional argument)
- `--start-year`: Start year (required)
- `--end-year`: End year (required)
- `--admin-level`: Administrative level (0=country, 1=state/province, 2=county/department, default: 1)
- `--variables`: Specific variables to process (e.g., lai_low lai_high, default: all available)
- `--output-format`: Output format (csv or parquet, default: csv)
- `--debug`: Enable debug logging

### 5. Process Soil Data

```bash
# Process all soil properties and depths for a country (default behavior)
python -m cli.process_soil --country USA

# Process specific properties and depths
python -m cli.process_soil --country Argentina --properties bulk_density clay --depths 0_5cm 5_15cm

# Process to state-level instead of county-level
python -m cli.process_soil --country Brazil --properties clay sand silt --depths 0_5cm 5_15cm --admin-level 1
```

**Available options:**
- `--country`: Country to process (required)
- `--properties`: Soil properties to process (default: all available, choices: bulk_density, cec, coarse_fragments, clay, nitrogen, organic_carbon_density, ph_h2o, sand, silt, organic_carbon)
- `--depths`: Depth ranges to process (default: all available, choices: 0_5cm, 5_15cm, 15_30cm, 30_60cm, 60_100cm, 100_200cm)
- `--admin-level`: Administrative level (0=country, 1=state/province, 2=county/department, default: 1)
- `--output-format`: Output format (csv or parquet, default: csv)
- `--debug`: Enable debug logging

## Project Structure

```
amazing-crop-yield-datasets/
├── cli/                        # Command-line interfaces
│   ├── __init__.py            # CLI package initialization
│   ├── __main__.py            # Unified CLI entry point
│   ├── download_weather.py    # Weather data download script
│   ├── download_land_surface.py # Land surface data download script
│   ├── download_soil.py       # Soil data download script
│   ├── process_weather.py     # Weather data processing script
│   ├── process_land_surface.py # Land surface data processing script
│   └── process_soil.py        # Soil data processing script
├── src/                       # Source code modules
│   ├── downloader/           # Data download modules
│   ├── processing/           # Data processing utilities
│   └── utils/               # Utility functions
├── data/                     # Data storage (created during processing)
└── requirements.txt          # Python dependencies
```

## Processing Pipeline

The processing pipeline follows a consistent approach across all data types: download raw data in its native format, standardize to NetCDF, then perform spatial aggregation to administrative boundaries.

### General Approach

1. **Download**: Data is downloaded in the format provided by the source (ZIP, TIFF, NetCDF) and saved to `data/{country}/{data_type}/` folders
2. **Standardization**: All data is converted to NetCDF format for consistent processing
3. **Spatial Aggregation**: Gridded data is aggregated to administrative boundary averages using different masking strategies depending on data type

### Data Flow Overview

```
Raw Data (ZIP/TIFF/NetCDF)
    ↓ [Format Standardization]
NetCDF Files
    ↓ [Spatial Aggregation with Masking]
Administrative-Level CSV Files
```

### Processing Strategies by Data Type

#### Weather Data Processing
- **Temporal Component**: Daily → Weekly aggregation (52 weeks per year)
- **Masking Strategy**: **Cropland mask + Admin boundaries**
- **Rationale**: Weather data is most relevant for crop production areas, so we apply HYDE cropland masks and compute area-weighted averages only over areas with >1% cropland coverage

#### Land Surface Data Processing  
- **Temporal Component**: Daily/Weekly time series
- **Masking Strategy**: **Admin boundaries only (no cropland mask)**
- **Rationale**: LAI and other vegetation indices vary dramatically throughout the growing season. Applying a static cropland mask would exclude many administrative areas during non-growing periods when LAI is naturally low
- **NDVI Dataset Handling**: Automatically switches between NOAA CDR AVHRR NDVI (1982-2013) and NOAA CDR VIIRS NDVI (2014+) to provide consistent temporal coverage

#### Soil Data Processing
- **Temporal Component**: None (static properties)
- **Masking Strategy**: **Admin boundaries only (no cropland mask)**
- **Rationale**: Soil properties are inherent characteristics of the landscape. Using any specific year's cropland mask would introduce temporal bias, as soil properties are relevant for potential agricultural use regardless of current land use

### Spatial Aggregation Process

#### Two-Stage Process:

**Stage 1: MASKING** - Determine which grid cells to include
- **Admin Boundary Mask**: Maps each grid cell to its administrative unit (cached globally)
- **Cropland Mask**: Applied only for weather data using HYDE 3.5 historical data
- **Combined Processing**: Weather data uses both masks; land surface and soil use admin boundaries only

**Stage 2: AGGREGATION** - Compute area-weighted averages
- **25-Subcell Subdivision**: Each grid cell divided into 5×5 subcells for precise area weighting
- **Vectorized Computation**: Processes all administrative units simultaneously
- **Memory Optimized**: Caches admin metadata to avoid expensive pandas operations

### Technical Implementation

#### Key Optimizations:
1. **Format Standardization**: All data converted to NetCDF for consistent processing
2. **Mask Caching**: Admin boundary and cropland masks cached as NetCDF files
3. **Area Weighting**: 25-subcell subdivision for accurate area calculations
4. **Vectorized Operations**: Batch processing of all admin units per time step
5. **Memory Management**: Chunk-based processing for large datasets

#### Data Specifications:
- **Grid Resolution**: 0.1° × 0.1° (typical for AgERA5/ERA5 data)
- **Administrative Levels**: 
  - Level 1: States/Provinces
  - Level 2: Counties/Departments (default)
- **Data Sources**:
  - **Boundaries**: GADM (Global Administrative Areas) database
  - **Cropland**: HYDE 3.5 historical land use dataset (weather data only)
- **Coordinate System**: WGS84 (EPSG:4326)

### Output Formatting

**Purpose**: Transform aggregated data into final CSV format
- **Weather**: Time series with weekly columns: `country,admin_level_1,admin_level_2,year,week_1,week_2,...,week_52`
- **Land Surface**: Time series with weekly/daily columns depending on temporal resolution
- **Soil**: Static values: `country,admin_level_1,admin_level_2,property_depth_value`


## Output

Creates CSV files like:
```
data/usa/intermediate/aggregated/weather_2020-2021_temp_min_weekly_weighted_admin2.csv
data/usa/intermediate/aggregated/weather_2020-2021_temp_max_weekly_weighted_admin2.csv
data/usa/intermediate/aggregated/weather_2020-2021_precip_weekly_weighted_admin2.csv
```

## Directory Structure

The project organizes data into three main categories:

```
data/{country}/
├── raw/                    # Downloaded raw data files
│   ├── weather/           # AgERA5 weather data (ZIP files)
│   ├── land_surface/      # ERA5 land surface data (TIF files)
│   ├── soil/              # SoilGrids soil data (TIF files)
│   ├── gadm/              # GADM administrative boundaries
│   ├── crop_yield/        # Raw crop yield data (CSV files)
│   ├── planted_area/      # Raw planted area data
│   └── faostat/           # Raw FAOSTAT data
├── intermediate/           # Processed intermediate files
│   ├── admin_mask/        # Administrative boundary masks
│   ├── cropland_mask/     # Cropland masks
│   ├── weather/           # Processed weather data
│   ├── land_surface/      # Processed land surface data
│   ├── soil/              # Processed soil data
│   └── aggregated/        # Aggregated CSV files
└── final/                 # Final processed datasets
    ├── crop_<cropname*>_yield.csv     # Processed crop yield data
    └── merged_data_chunk_*.csv    # Final merged datasets
```

Each file contains columns:
```
country,admin_level_1,admin_level_2,year,week_1,week_2,...,week_52
```

Ready for machine learning with 52 weekly columns per weather variable.

## Crop Yield Data

The repository includes crop yield data processing for Argentina, with plans for additional countries.

### Argentina Crop Yield Processing

Process raw Argentina crop yield data from MAGyP into standardized CSV format:

```bash
# Process all Argentina crop yield data
python -m src.crop_yield.argentina.processor
```
For detailed instructions, see: [`src/crop_yield/argentina/readme.md`](src/crop_yield/argentina/readme.md)

## Future Extensions

The CLI structure is designed to easily accommodate additional data types:
- Additional crop yield datasets (USA, Brazil, etc.)
- Satellite imagery processing
- Economic indicators
- etc.
