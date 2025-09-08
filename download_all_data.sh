#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -J crop_download

# Default values
START_YEAR=1979
END_YEAR=2025
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
        --country)
            COUNTRY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch download_all_data.sh --country COUNTRY [--start-year YEAR] [--end-year YEAR]"
            echo "Example: sbatch download_all_data.sh --country Argentina --start-year 1980 --end-year 2020"
            exit 1
            ;;
    esac
done

# Check if country is provided
if [ -z "$COUNTRY" ]; then
    echo "Error: --country is required"
    echo "Usage: sbatch download_all_data.sh --country COUNTRY [--start-year YEAR] [--end-year YEAR]"
    echo "Example: sbatch download_all_data.sh --country Argentina --start-year 1980 --end-year 2020"
    exit 1
fi

NDVI_START_YEAR=$((START_YEAR > 1982 ? START_YEAR : 1982))

# Load required modules
module load miniforge/24.3.0-0

# Navigate to the project directory
cd /home/notadib/projects/amazing-crop-yield-datasets

# Set up Python environment if needed
# conda activate your_env_name  # uncomment if using conda environment
# pip install -r requirements.txt  # uncomment if dependencies not installed

# Download weather data
echo "Starting weather data download for $COUNTRY..."
python -m cli.download_weather --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --concurrent 8

# Download land surface data (requires Google Earth Engine authentication)
echo "Starting land surface data download for $COUNTRY..."
python -m cli.download_land_surface --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --concurrent 8 --variables lai_low lai_high
# ndvi doesnt exist between 1979-81 so need to download separately
python -m cli.download_land_surface --country "$COUNTRY" --start-year $NDVI_START_YEAR --end-year $END_YEAR --concurrent 8 --variables ndvi

# Download soil data (no temporal component)
echo "Starting soil data download for $COUNTRY..."
python -m cli.download_soil --country "$COUNTRY" --concurrent 2

echo "All downloads completed for $COUNTRY"
