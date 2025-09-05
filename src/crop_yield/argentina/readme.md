# Argentina Department-Level Crop Yield Data

**Source:** Ministerio de Agricultura, Ganadería y Pesca (MAGyP) – *Estimaciones Agrícolas* portal.  
URL: [https://datosestimaciones.magyp.gob.ar/reportes.php?reporte=Estimaciones](https://datosestimaciones.magyp.gob.ar/reportes.php?reporte=Estimaciones)

---

## Data Download Guide

### Steps to Download Department-Level Data

1. **Open the Portal**
   - Go to the URL above.  
   - The interface is in Spanish.

2. **Select Crop (Cultivo)**
   - wheat = Trigo total
   - soybean = Soja total
   - corn = Maíz
   - sunflower = Girasol

3. **Select Variable (Variable)**
   - Choose **Rendimiento** = Yield (kg per hectare).  
   - Other options: *Producción* (Production), *Superficie sembrada* (Area planted).

4. **Set Years**
   - **Desde** = From year  
   - **Hasta** = To year  
   - Data typically available from 1969/1970 onwards (depends on crop).

5. **Set Aggregation (Agregación)**
   - Choose **Total Departamentos** = Department level.  
   - Other options: Provincias (province), Partidos (district), etc.

6. **Generate Report**
   - Click **Consultar** (Search/Query).  
   - A table will appear below with yearly values.

7. **Download CSV**
   - Click **Descargar** = Download.  
   - This saves a CSV file of your query.
   - Save it as `data/argentina/crop_yield/<crop_name>.csv`

### Example: Wheat in Córdoba Departments
- Cultivo: `Trigo total`  
- Variable: `Rendimiento`  
- Desde: `1970/71`  
- Hasta: `2024/25`  
- Agregación: `Total Departamentos`  
- Then click **Consultar → Descargar**.
- Save as `data/argentina/crop_yield/wheat.csv`

---

## Data Processing

After downloading the raw CSV files, use the processing pipeline to convert them into standardized format.

### Processing Pipeline

The processing system includes:

1. **`models.py`** - Configuration and mappings
   - Spanish to English crop name mapping
   - Data quality thresholds (80% completeness required)
   - Column name mappings

2. **`processor.py`** - Main processing logic
   - Loads raw CSV files with Spanish headers
   - Extracts harvest year from season strings (e.g., "2023/24" → 2024)
   - Filters departments with insufficient data quality
   - Outputs standardized CSV files

### Usage

#### Command Line
```bash
# Process all crops (clean output)
python -m src.crop_yield.argentina.processor

# Process specific crops
python -m src.crop_yield.argentina.processor --crop wheat corn soybean

# Process single crop with verbose output
python -m src.crop_yield.argentina.processor --crop wheat --debug

# Show help
python -m src.crop_yield.argentina.processor --help
```

**Available crops:** wheat, soybean, corn, sunflower, barley

### Output Format

Processed files are saved as: `{crop_name}_yield_{min_year}-{max_year}.csv`

**Output columns:**
- `country` - Always "Argentina"
- `admin_level_1` - Province name
- `admin_level_2` - Department name  
- `year` - Harvest year (extracted from season)
- `{crop_name}_yield` - Yield in kg/hectare

**Example output files:**
- `crop_wheat_yield_1970-2025.csv`
- `crop_corn_yield_1970-2025.csv`
- `crop_soybean_yield_1975-2025.csv`
- `crop_sunflower_yield_1970-2025.csv`
- `crop_barley_yield_2017-2024.csv`

### Data Quality Filtering

The processor applies quality filtering:

- Departments with >50% missing data in the last 20 years are dropped entirely

### Statistics Output

The processor provides comprehensive statistics:

- **Department filtering**: How many departments were kept/dropped
- **Data completeness**: Mean, min, max completeness rates
- **Missing data**: Total records, missing percentages, year ranges
- **Quality metrics**: Worst year missing percentage, average missing per year

**Example statistics output:**
```
WHEAT
----
Departments:
  Total: 245
  Kept: 198 (47 dropped, 19.2%)
  Completeness (last 10 years):
    Mean: 89.5%
    Min: 60.0%
    Max: 100.0%
Missing Data:
  Total records: 8,945
  Missing yield: 487 (5.4%)
  Year range: 1970-2025
  Years with data: 56/56
  Worst year missing: 15.2%
  Average missing per year: 5.4%
```