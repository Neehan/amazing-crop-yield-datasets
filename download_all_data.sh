#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -J crop_download

# Check if country argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: sbatch download_all_data.sh <COUNTRY>"
    echo "Example: sbatch download_all_data.sh Argentina"
    exit 1
fi

COUNTRY="$1"
START_YEAR=1979
END_YEAR=2025

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
python -m cli.download_land_surface --country "$COUNTRY" --start-year $START_YEAR --end-year $END_YEAR --concurrent 4

# Download soil data (no temporal component)
echo "Starting soil data download for $COUNTRY..."
python -m cli.download_soil --country "$COUNTRY" --concurrent 5

echo "All downloads completed for $COUNTRY"
