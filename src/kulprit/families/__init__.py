"""Distribution families module."""

from abc import ABC, abstractmethod


class BaseFamily(ABC):
    """Base family class."""

    @abstractmethod
    def solve_analytic(self):
        pass

    @abstractmethod
    def solve_dispersion(self):
        pass
