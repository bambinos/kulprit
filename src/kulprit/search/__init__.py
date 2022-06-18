"""Base search module."""


from abc import ABC, abstractmethod


class SearchPath(ABC):
    """Base search module."""

    @abstractmethod
    def search(self, max_terms: int) -> None:
        """Initialise search path class."""
