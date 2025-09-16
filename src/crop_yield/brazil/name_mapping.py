"""Brazil-specific administrative name mapping for crop yield data"""

from typing import Dict
from src.crop_yield.base.name_mapping import BaseNameMapper


class BrazilNameMapper(BaseNameMapper):
    """Brazil-specific name mapper"""

    def get_exceptions(self, admin_level: int) -> Dict[str, str]:
        """Exceptions that can't be handled by normalization"""
        if admin_level == 1:
            # Map Brazil 2-letter state codes to GADM state names
            return {
                "AC": "Acre",
                "AL": "Alagoas",
                "AM": "Amazonas",
                "AP": "Amapá",
                "BA": "Bahia",
                "CE": "Ceará",
                "DF": "Distrito Federal",
                "ES": "Espírito Santo",
                "GO": "Goiás",
                "MA": "Maranhão",
                "MG": "Minas Gerais",
                "MS": "Mato Grosso do Sul",
                "MT": "Mato Grosso",
                "PA": "Pará",
                "PB": "Paraíba",
                "PE": "Pernambuco",
                "PI": "Piauí",
                "PR": "Paraná",
                "RJ": "Rio de Janeiro",
                "RN": "Rio Grande do Norte",
                "RO": "Rondônia",
                "RR": "Roraima",
                "RS": "Rio Grande do Sul",
                "SC": "Santa Catarina",
                "SE": "Sergipe",
                "SP": "São Paulo",
                "TO": "Tocantins",
            }
        elif admin_level == 2:
            # Map specific admin level 2 names that don't match GADM exactly
            return {
                "Graccho Cardoso": "Gracho Cardoso",
                "Barão do Monte Alto": "Barão de Monte Alto",
                "Santo Antônio de Leverger": "Santo Antônio do Leverger",
            }
        return {}
