"""
Models and configuration for Brazil crop yield data processing.

This module contains mappings between Portuguese crop names and standardized English names,
as well as IBGE crop codes and data quality thresholds.
"""

from typing import Dict, Set

# IBGE crop codes mapping
CROP_CODES: Dict[str, str] = {
    "corn": "2711",
    "wheat": "2716",
    "soybean": "2713",
    "rice": "2692",
    "beans": "2702",
    "sugarcane": "2696",
    "cotton": "2689",
    "sunflower": "109179",
    "sorghum": "2714",
    "oats": "2693",
    "barley": "2699",
    "rye": "2698",
    "triticale": "109180",
    "potato": "2695",
    "sweet_potato": "2694",
    "cassava": "2708",
    "tomato": "2715",
    "onion": "2697",
    "garlic": "2690",
    "peanut": "2691",
    "tobacco": "2703",
    "watermelon": "2709",
    "melon": "2710",
    "pineapple": "2688",
    "castor": "2707",
    "jute": "2704",
    "flax": "2705",
    "ramie": "2712",
    "mallow": "2706",
    "pea": "2700",
    "fava": "2701",
    "alfalfa": "40471",
    "sugar_cane_forage": "40470",
}

# Mapping from Portuguese crop names (as they appear in the data) to standardized English names
CROP_NAME_MAPPING: Dict[str, str] = {
    "Milho (em grão)": "corn",
    "Trigo (em grão)": "wheat",
    "Soja (em grão)": "soybean",
    "Arroz (em casca)": "rice",
    "Feijão (em grão)": "beans",
    "Cana-de-açúcar": "sugarcane",
    "Algodão herbáceo (em caroço)": "cotton",
    "Girassol (em grão)": "sunflower",
    "Sorgo (em grão)": "sorghum",
    "Aveia (em grão)": "oats",
    "Cevada (em grão)": "barley",
    "Centeio (em grão)": "rye",
    "Triticale (em grão)": "triticale",
    "Batata-inglesa": "potato",
    "Batata-doce": "sweet_potato",
    "Mandioca": "cassava",
    "Tomate": "tomato",
    "Cebola": "onion",
    "Alho": "garlic",
    "Amendoim (em casca)": "peanut",
    "Fumo (em folha)": "tobacco",
    "Melancia": "watermelon",
    "Melão": "melon",
    "Abacaxi": "pineapple",
    "Mamona (baga)": "castor",
    "Juta (fibra)": "jute",
    "Linho (semente)": "flax",
    "Rami (fibra)": "ramie",
    "Malva (fibra)": "mallow",
    "Ervilha (em grão)": "pea",
    "Fava (em grão)": "fava",
    "Alfafa fenada": "alfalfa",
    "Cana para forragem": "sugar_cane_forage",
}

# Set of all supported crops for validation
SUPPORTED_CROPS: Set[str] = set(CROP_NAME_MAPPING.keys())


# Column names in the raw data
RAW_DATA_COLUMNS = {
    "municipality_code": "municipality_code",
    "municipality": "municipality",
    "year": "year",
    "crop_code": "crop_code",
    "crop": "crop",
    "yield": "yield_kg_ha",
}

# Output CSV column names
OUTPUT_COLUMNS = [
    "country",
    "admin_level_1",
    "admin_level_2",
    "year",
    # crop yield columns will be added dynamically: {crop_name}_yield
]

# Constants
COUNTRY_NAME = "Brazil"
