"""Losses module."""

import torch.nn as nn


class Loss(nn.Module):
    """Base loss class."""

    def __init__(self):
        super(Loss, self).__init__()
