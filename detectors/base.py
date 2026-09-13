from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from schemas import Evidence


class BaseDetector(ABC):
    """Abstract base class for all detection lanes."""

    @abstractmethod
    def analyse(self, input_data: Any, metadata: Optional[Dict[str, Any]] = None) -> Evidence:
        """
        Analyze input data and return a structured Evidence object.
        """
        pass
