"""
Kulprit.

Kullback-Leibler projections for Bayesian model selection.
"""

from kulprit.projector import ProjectionPredictive
from kulprit.plots import plot_compare, plot_dist, plot_forest

__version__ = "0.5.1"


__all__ = ["ProjectionPredictive", "plot_compare", "plot_dist", "plot_forest"]
