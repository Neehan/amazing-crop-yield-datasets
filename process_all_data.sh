#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 48
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J crop_process

# Default values
START_YEAR=1979
END_YEAR=2025
ADMIN_LEVEL=2  # County/department level (default)
COUNTRY="argentina"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --end-year)
            END_YEAR="$2"
            shift 2
            ;;
        --admin-level)
            ADMIN_LEVEL="$2"
            shift 2
            ;;
        --country)
            COUNTRY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch process_all_data.sh --country COUNTRY [--start-year YEAR] [--end-year YEAR] [--admin-level LEVEL]"
            echo "Example: sbatch process_all_data.sh --country Argentina --start-year 1980 --end-year 2020 --admin-level 2"
            exit 1
            ;;
    esac
done

# Check if country is provided
if [ -z "$COUNTRY" ]; then
    echo "Error: --country is required"
    echo "Usage: sbatch process_all_data.sh --country COUNTRY [--start-year YEAR] [--end-year YEAR] [--admin-level LEVEL]"
    echo "Example: sbatch process_all_data.sh --country Argentina --start-year 1980 --end-year 2020 --admin-level 1"
    exit 1
fi

NDVI_START_YEAR=$((START_YEAR > 1982 ? START_YEAR : 1982))

# Load required modules
module load miniforge/24.3.0-0

# Navigate to the project directory
cd /home/notadib/projects/amazing-crop-yield-datasets

# Set up Python environment if needed
# conda activate your_env_name  # uncomment if using conda environment

echo "Starting data processing for $COUNTRY..."
echo "Years: $START_YEAR to $END_YEAR"
echo "Admin level: $ADMIN_LEVEL"

# Process weather data
echo "Starting weather data processing for $COUNTRY..."
python -m cli.process_weather --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --admin-level $ADMIN_LEVEL

# Process land surface data
echo "Starting land surface data processing for $COUNTRY..."
# Process LAI data (available from 1979)
python -m cli.process_land_surface --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --admin-level $ADMIN_LEVEL --variables lai_low lai_high
# Process NDVI data (only available from 1982)
python -m cli.process_land_surface --country "$COUNTRY" --start-year $NDVI_START_YEAR --end-year $END_YEAR --admin-level $ADMIN_LEVEL --variables ndvi

# Process soil data (no temporal component, so no start/end year needed)
echo "Starting soil data processing for $COUNTRY..."
python -m cli.process_soil --country "$COUNTRY" --admin-level $ADMIN_LEVEL

# Process final merged dataset
echo "Starting final dataset aggregation for $COUNTRY..."
python -m cli.process_datasaver --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --admin-level $ADMIN_LEVEL --chunk-size 50

# Process crop calendar data
echo "Starting crop calendar processing for $COUNTRY..."
python -m cli.process_crop_calendar --country "$COUNTRY" --admin-level $ADMIN_LEVEL

# Process irrigation data
echo "Starting irrigation data processing for $COUNTRY..."
python -m cli.process_irrigation --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --admin-level $ADMIN_LEVEL

echo "All processing completed for $COUNTRY"
echo "Intermediate files are in data/$COUNTRY/intermediate/"
echo "Final merged feature datasets are in data/$COUNTRY/final/features/"
