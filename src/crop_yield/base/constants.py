"""
Common constants for crop yield data processing.
"""

# Data quality thresholds
DATA_QUALITY_THRESHOLD = 0.25  # 25% of data must be present (75% missing allowed)
EVALUATION_YEARS = 20  # Evaluate data quality over the last 20 years

# Generic column names used across all countries
AREA_PLANTED_COLUMN = "area_planted"
AREA_HARVESTED_COLUMN = "area_harvested"
PRODUCTION_COLUMN = "production"
YIELD_COLUMN = "yield"
