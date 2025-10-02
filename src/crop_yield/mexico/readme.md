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

**Irrigation Modality Variants:**
- **Default (no suffix):** Combined irrigated + rainfed/temporal data (modality 3)
- **`_irrigated` suffix:** Irrigated-only data (modality 1)  
  Examples: `beans_irrigated`, `corn_irrigated`, `wheat_irrigated`
- **`_rainfed` suffix:** Rainfed/temporal-only data (modality 2)  
  Examples: `beans_rainfed`, `corn_rainfed`, `wheat_rainfed`

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

# Download irrigated-only data
python -m src.crop_yield.mexico.downloader --crops beans_irrigated corn_irrigated

# Download rainfed-only data
python -m src.crop_yield.mexico.downloader --crops beans_rainfed corn_rainfed wheat_rainfed

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

# Process irrigated or rainfed variants
python -m src.crop_yield.mexico.processor --crops beans_irrigated corn_rainfed

# Process specific crops with custom year range
python -m src.crop_yield.mexico.processor --crops tomato beans --start-year 2020 --end-year 2025
```

## Output

Processed files: `crop_{crop_name}_yield.csv` (one file per crop)

**Example output files:**
- `crop_corn_yield.csv` (50,835 records) - combined irrigated + rainfed
- `crop_beans_yield.csv` (37,902 records) - combined irrigated + rainfed
- `crop_tomato_yield.csv` (15,391 records) - combined irrigated + rainfed
- `crop_sorghum_yield.csv` (12,246 records) - combined irrigated + rainfed
- `crop_wheat_yield.csv` (10,387 records) - combined irrigated + rainfed
- `crop_sugarcane_yield.csv` (5,531 records) - combined irrigated + rainfed
- `crop_barley_yield.csv` (4,541 records) - combined irrigated + rainfed
- `crop_soybean_yield.csv` (685 records) - combined irrigated + rainfed

**Irrigation modality variants:**
- `crop_beans_irrigated_yield.csv` - irrigated-only data
- `crop_beans_rainfed_yield.csv` - rainfed/temporal-only data
- Similar files can be created for any crop by adding `_irrigated` or `_rainfed` suffix

**Columns:**
- `country` - "Mexico"
- `admin_level_1` - State name (GADM compatible)
- `admin_level_2` - Municipality name (GADM compatible)
- `year` - Year
- `{crop_name}_yield` - Yield in kg/ha (e.g., `beans_yield`, `beans_irrigated_yield`, `beans_rainfed_yield`)
- `area_planted` - Area planted in hectares (Superficie Sembrada)
- `area_harvested` - Area harvested in hectares (Superficie Cosechada)
- `production` - Production quantity in tonnes (Producción)

