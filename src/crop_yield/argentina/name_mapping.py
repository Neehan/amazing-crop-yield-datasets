"""Argentina-specific administrative name mapping for crop yield data"""

from typing import Dict
from src.crop_yield.base.name_mapping import BaseNameMapper


class ArgentinaNameMapper(BaseNameMapper):
    """Argentina-specific name mapper"""

    def get_exceptions(self, admin_level: int) -> Dict[str, str]:
        """Exceptions that can't be handled by normalization"""
        if admin_level == 1:
            return {}
        elif admin_level == 2:
            return {
                # Number words that normalization can't handle
                "25 DE MAYO": "Veinticinco de Mayo",
                "9 DE JULIO": "Nueve de Julio",
                "12 DE OCTUBRE": "Doce de Octubre",
                "1 DE MAYO": "Primero de Mayo", 
                
                # Abbreviations that normalization can't handle
                "GRAL. ALVEAR": "General Alvear",
                "CNEL. SUAREZ": "Coronel Suárez", 
                "CORONEL DE MARINA L ROSALES": "Coronel de Marina Leonardo Rosal",
                "DR. MANUEL BELGRANO": "General Manuel Belgrano",
                "JUAN MARTIN DE PUEYRREDON": "General Pueyrredón",
                "JUAN F. IBARRA": "Juan Felipe Ibarra",
                
                # Names that don't exist in GADM - keep original
                "PUNTA INDIO": "Punta Indio",
                "2 DE ABRIL": "2 de Abril",
                "SAN SALVADOR": "San Salvador",
                "SIN DEFINIR": "Sin Definir",
            }
        else:
            return {}