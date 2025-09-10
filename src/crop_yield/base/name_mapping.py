"""Base administrative name mapping for standardizing crop yield data with GADM boundaries"""

import logging
import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, Set

logger = logging.getLogger(__name__)


class BaseNameMapper(ABC):
    """Base class for mapping administrative names from national statistics to GADM standard names"""

    def __init__(self, gadm_admin1: Set[str], gadm_admin2: Set[str]):
        """Initialize mapper with GADM names"""
        # Build lookup: admin_level -> {normalized_name -> original_gadm_name}
        self.admin_lookup = {1: {}, 2: {}}
        self.unknown_names = set()

        for name in gadm_admin1:
            self.admin_lookup[1][self._normalize_name(name)] = name

        for name in gadm_admin2:
            self.admin_lookup[2][self._normalize_name(name)] = name

    def _normalize_name(self, name: str) -> str:
        """Remove accents and convert to uppercase"""
        normalized = unicodedata.normalize("NFD", name)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        return normalized.upper()

    @abstractmethod
    def get_exceptions(self, admin_level: int) -> Dict[str, str]:
        """Return mapping of raw names to GADM names for exceptions"""
        pass

    def map_admin_name(self, raw_name: str, admin_level: int) -> str:
        """Map raw admin name to GADM name"""
        # Check exceptions first
        exceptions = self.get_exceptions(admin_level)
        if raw_name in exceptions:
            return exceptions[raw_name]

        # Normalize and lookup
        normalized = self._normalize_name(raw_name)
        result = self.admin_lookup[admin_level].get(normalized)
        if result:
            return result

        # If can't map, return normalized original name
        if raw_name not in self.unknown_names:
            self.unknown_names.add(raw_name)
            logger.warning(f"Cannot map admin level {admin_level}: {raw_name}")
        return normalized.title()
