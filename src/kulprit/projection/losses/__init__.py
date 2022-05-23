"""Losses module."""

from abc import ABC, abstractmethod

import torch.nn as nn


class Loss(nn.Module, ABC):
    """Base loss class."""

    @abstractmethod
    def forward(self):  # pragma: no cover
        pass
