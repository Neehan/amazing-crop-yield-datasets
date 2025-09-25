# Mexico Municipality-Level Crop Yield Data

**Source:** Sistema de Información Agroalimentaria y Pesquera (SIAP) – *Cierre Agrícola*  
URL: [https://nube.agricultura.gob.mx/cierre_agricola/](https://nube.agricultura.gob.mx/cierre_agricola/)

## Available Crops
* **corn** (Maíz grano) - Code: 225
* **soybean** (Soya) - Code: 375  
* **wheat** (Trigo grano) - Code: 395
* **sorghum** (Sorgo grano) - Code: 374
* **sugarcane** (Caña de azúcar) - Code: 70
* **tomato** (Tomate rojo) - Code: 389
* **beans** (Frijol) - Code: 152
* **barley** (Cebada grano) - Code: 77

## Prerequisites

Before processing data, download the GADM administrative boundaries:

```bash
# Create GADM directory
mkdir -p data/mexico/gadm

# Download Mexico GADM data (required for name mapping)
wget -O data/mexico/gadm/gadm41_MEX.gpkg "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_MEX.gpkg"
```

## Download

```bash
# Download ALL crops (default behavior - downloads all 8 crops)
python -m src.crop_yield.mexico.downloader

# Download specific crops
python -m src.crop_yield.mexico.downloader --crops corn soybean wheat

# Download specific year range for all crops (2020, 2021 and 2022)
python -m src.crop_yield.mexico.downloader --start-year 2020 --end-year 2023

# Download with debug output
python -m src.crop_yield.mexico.downloader --debug

# Download specific crops with custom year range
python -m src.crop_yield.mexico.downloader --crops corn tomato --start-year 2020 --end-year 2025
```

## Process

```bash
# Process ALL crops (default behavior - processes all 8 crops)
python -m src.crop_yield.mexico.processor

# Process specific crops
python -m src.crop_yield.mexico.processor --crops corn soybean wheat

# Process specific year range for all crops
python -m src.crop_yield.mexico.processor --start-year 2020 --end-year 2023

# Process with debug output
python -m src.crop_yield.mexico.processor --debug

# Process specific crops with custom year range
python -m src.crop_yield.mexico.processor --crops tomato beans --start-year 2020 --end-year 2025
```

## Output

Processed files: `crop_{crop_name}_yield.csv` (one file per crop)

**Example output files:**
- `crop_corn_yield.csv` (50,835 records)
- `crop_beans_yield.csv` (37,902 records)  
- `crop_tomato_yield.csv` (15,391 records)
- `crop_sorghum_yield.csv` (12,246 records)
- `crop_wheat_yield.csv` (10,387 records)
- `crop_sugarcane_yield.csv` (5,531 records)
- `crop_barley_yield.csv` (4,541 records)
- `crop_soybean_yield.csv` (685 records)

**Columns:**
- `country` - "Mexico"
- `admin_level_1` - State name (GADM compatible)
- `admin_level_2` - Municipality name (GADM compatible)
- `year` - Year
- `yield` - Yield in kg/ha (converted from tonnes/ha)
- `area_planted` - Area planted in hectares (Superficie Sembrada)
- `area_harvested` - Area harvested in hectares (Superficie Cosechada)
- `production` - Production quantity in tonnes (Producción)

