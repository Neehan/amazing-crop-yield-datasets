"""Crop calendar processing configuration"""

from pathlib import Path
from typing import List, Optional

from src.processing.base.config import ProcessingConfig

# MIRCA2000 crop codes mapping
CROP_CODES = {
    1: "wheat",
    2: "corn",
    3: "rice",
    4: "barley",
    5: "rye",
    6: "millet",
    7: "sorghum",
    8: "soybean",
    9: "sunflower",
    10: "potatoes",
    11: "cassava",
    12: "sugar_cane",
    13: "sugar_beet",
    14: "oil_palm",
    15: "rapeseed_canola",
    16: "groundnuts_peanuts",
    17: "pulses",
    18: "citrus",
    19: "date_palm",
    20: "grapes_vine",
    21: "cotton",
    22: "cocoa",
    23: "coffee",
    24: "others_perennial",
    25: "fodder_grasses",
    26: "others_annual",
    27: "wheat",
    28: "corn",
    29: "rice",
    30: "barley",
    31: "rye",
    32: "millet",
    33: "sorghum",
    34: "soybean",
    35: "sunflower",
    36: "potatoes",
    37: "cassava",
    38: "sugar_cane",
    39: "sugar_beet",
    40: "oil_palm",
    41: "rapeseed_canola",
    42: "groundnuts_peanuts",
    43: "pulses",
    44: "citrus",
    45: "date_palm",
    46: "grapes_vine",
    47: "cotton",
    48: "cocoa",
    49: "coffee",
    50: "others_perennial",
    51: "fodder_grasses",
    52: "others_annual",
}

# Reverse mapping for crop names to codes
CROP_NAME_TO_CODES = {}
for code, name in CROP_CODES.items():
    if name not in CROP_NAME_TO_CODES:
        CROP_NAME_TO_CODES[name] = []
    CROP_NAME_TO_CODES[name].append(code)

# Default crops to process (most common ones)
DEFAULT_CROP_NAMES = ["wheat", "corn", "rice", "soybean"]


class CropCalendarConfig(ProcessingConfig):
    """Configuration for crop calendar data processing"""

    def __init__(
        self,
        country: str,
        crop_names: Optional[List[str]],
        admin_level: int,
        data_dir: Path,
        output_format: str,
        debug: bool,
    ):
        super().__init__(country, admin_level, data_dir, output_format, debug)
        self.crop_names = crop_names or self._auto_detect_crops()
        # Convert crop names to codes internally
        self.crop_codes = self._convert_names_to_codes(self.crop_names)

    def _auto_detect_crops(self) -> List[str]:
        """Auto-detect crops from yield data in final directory

        Returns:
            List of crop names found in final directory
        """
        final_dir = self.data_dir / self.country.lower() / "final"
        if not final_dir.exists():
            return DEFAULT_CROP_NAMES  # Default fallback

        crop_files = list(final_dir.glob("crop_*_yield_*.csv"))
        crops = []

        for file_path in crop_files:
            # Extract crop name from filename like "crop_wheat_yield_1970-2025.csv"
            filename = file_path.stem
            if filename.startswith("crop_") and "_yield_" in filename:
                crop_name = filename.split("_yield_")[0].replace("crop_", "")
                # Convert underscores to single words (e.g., "sugar_cane" -> "sugarcane")
                crop_name = crop_name.replace("_", "")

                # Check if crop name is valid
                if crop_name in CROP_NAME_TO_CODES:
                    crops.append(crop_name)

        if not crops:
            return DEFAULT_CROP_NAMES  # Default fallback

        return sorted(list(set(crops)))  # Remove duplicates and sort

    def validate(self) -> None:
        """Validate crop calendar specific configuration"""
        super().validate()

        valid_crop_names = list(CROP_NAME_TO_CODES.keys())
        invalid_crops = [c for c in self.crop_names if c not in valid_crop_names]
        if invalid_crops:
            raise ValueError(
                f"Invalid crop names: {invalid_crops}. Valid options: {valid_crop_names}"
            )

    def _convert_names_to_codes(self, crop_names: List[str]) -> List[int]:
        """Convert crop names to MIRCA codes (only irrigated codes, rainfed handled in gz_converter)"""
        codes = []
        for name in crop_names:
            if name in CROP_NAME_TO_CODES:
                # Only get irrigated codes (1-26), rainfed will be handled automatically
                irrigated_codes = [
                    code for code in CROP_NAME_TO_CODES[name] if code <= 26
                ]
                codes.extend(irrigated_codes)
        return codes

    def get_mirca_directory(self) -> Path:
        """Get MIRCA2000 data directory"""
        mirca_dir = self.data_dir / "global" / "mirca2000-v1.1"
        mirca_dir.mkdir(parents=True, exist_ok=True)
        return mirca_dir
