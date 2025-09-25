# Brazil Municipality-Level Crop Yield Data

**Source:** Instituto Brasileiro de Geografia e Estatística (IBGE) – *Produção Agrícola Municipal* (PAM) table 1612.  
URL: [https://sidra.ibge.gov.br/tabela/1612](https://sidra.ibge.gov.br/tabela/1612)

## Prerequisites

Before processing data, download the GADM administrative boundaries:

```bash
# Create GADM directory
mkdir -p data/brazil/gadm

# Download Brazil GADM data (required for name mapping)
wget -O data/brazil/gadm/gadm41_BRA.gpkg "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_BRA.gpkg"
```

## Download

```bash
# Download multiple years (default: 1974-2024)
python -m src.crop_yield.brazil.downloader --crop corn

# Download specific year range (2020, 2021 and 2022)
python -m src.crop_yield.brazil.downloader --crop corn --start-year 2020 --end-year 2023

# Download with debug output
python -m src.crop_yield.brazil.downloader --crop wheat --debug
```

**Available crops:** corn, wheat, soybean, rice, beans, sugarcane, cotton, sunflower, sorghum, oats, barley, rye, triticale, potato, sweet_potato, cassava, tomato, onion, garlic, peanut, tobacco, watermelon, melon, pineapple, castor, jute, flax, ramie, mallow, pea, fava, alfalfa, sugar_cane_forage

## Process

```bash
# Process all years (default: 1974-2024)
python -m src.crop_yield.brazil.processor --crop corn

# Process specific year range
python -m src.crop_yield.brazil.processor --crop corn --start-year 2020 --end-year 2023

# Process with debug output
python -m src.crop_yield.brazil.processor --crop wheat --debug
```

## Output

Processed files: `crop_{crop_name}_yield.csv`

**Columns:**
- `country` - "Brazil"
- `admin_level_1` - State name (GADM compatible)
- `admin_level_2` - Municipality name (GADM compatible)
- `year` - Year
- `{crop_name}_yield` - Yield in kg/ha
- `area_planted` - Area planted in hectares 
- `area_harvested` - Area harvested in hectares 
- `production` - Production quantity in tonnes 
**Features:**
- Filters zero and missing yields
- Maps 2-letter state codes to full names
- Handles municipality name variations for GADM compatibility
- Parallel downloads with progress bar
