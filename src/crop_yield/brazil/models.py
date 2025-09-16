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
    "soybean": "2702",
    "rice": "2701",
    "beans": "2703",
    "sugarcane": "2704",
    "cotton": "2705",
    "sunflower": "2706",
    "sorghum": "2707",
    "oats": "2708",
    "barley": "2709",
    "rye": "2710",
    "triticale": "2712",
    "potato": "2713",
    "sweet_potato": "2714",
    "cassava": "2715",
    "tomato": "2717",
    "onion": "2718",
    "garlic": "2719",
    "peanut": "2720",
    "tobacco": "2721",
    "watermelon": "2722",
    "melon": "2723",
    "pineapple": "2724",
    "castor": "2725",
    "jute": "2726",
    "flax": "2727",
    "ramie": "2728",
    "mallow": "2729",
    "pea": "2730",
    "fava": "2731",
    "alfalfa": "2732",
    "sugar_cane_forage": "2733",
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
