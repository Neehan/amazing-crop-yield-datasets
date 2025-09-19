#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH -c 48
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J to_khaki

# Default values
COUNTRY=""
CROPS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --country)
            COUNTRY="$2"
            shift 2
            ;;
        --crops)
            CROPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch khaki_format.sh --country COUNTRY --crops 'CROP1 CROP2 ...'"
            echo "Example: sbatch khaki_format.sh --country argentina --crops 'soybean corn wheat'"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$COUNTRY" ]; then
    echo "Error: --country is required"
    echo "Usage: sbatch khaki_format.sh --country COUNTRY --crops 'CROP1 CROP2 ...'"
    echo "Example: sbatch khaki_format.sh --country argentina --crops 'soybean corn wheat'"
    exit 1
fi

if [ -z "$CROPS" ]; then
    echo "Error: --crops is required"
    echo "Usage: sbatch khaki_format.sh --country COUNTRY --crops 'CROP1 CROP2 ...'"
    echo "Example: sbatch khaki_format.sh --country argentina --crops 'soybean corn wheat'"
    exit 1
fi

# Load required modules
module load miniforge/24.3.0-0

# Navigate to the project directory
cd /home/notadib/projects/amazing-crop-yield-datasets

# Set up Python environment if needed
# conda activate your_env_name  # uncomment if using conda environment

echo "Starting Khaki format conversion for $COUNTRY..."
echo "Crops: $CROPS"

# Convert to Khaki format
echo "Converting processed data to Khaki format for $COUNTRY..."
python convert_to_khaki_format.py --country "$COUNTRY" --crops $CROPS

echo "Khaki format conversion completed for $COUNTRY"
echo "Output file is in data/khaki/"