"""Base optimisation solvers module."""

from abc import ABC, abstractmethod


class BaseSolver(ABC):
    """Base solver class."""

    @abstractmethod
    def solve(self):  # pragma: no cover
        pass
