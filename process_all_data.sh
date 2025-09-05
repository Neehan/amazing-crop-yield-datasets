#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J crop_process

# Check if country argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: sbatch process_all_data.sh <COUNTRY>"
    echo "Example: sbatch process_all_data.sh Argentina"
    exit 1
fi

COUNTRY="$1"
START_YEAR=1979
END_YEAR=2025
ADMIN_LEVEL=2  # County/department level (default)

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
python -m cli.process_land_surface --country "$COUNTRY" --start-year 1982 --end-year $END_YEAR --admin-level $ADMIN_LEVEL --variables ndvi

# Process soil data (no temporal component, so no start/end year needed)
echo "Starting soil data processing for $COUNTRY..."
python -m cli.process_soil --country "$COUNTRY" --admin-level $ADMIN_LEVEL

echo "All processing completed for $COUNTRY"
echo "Output files are in data/$COUNTRY/processed/"
