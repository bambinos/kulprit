"""Projection class."""

import numpy as np
import pandas as pd

import torch

import arviz as az
import matplotlib.pyplot as plt

from .submodel import SubModel, KLDivSurrogateLoss


class Projector:
    def __init__(
        self, model, posterior, n_iters=200, lr=0.01, device=torch.device("cpu")
    ):
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Args:
            model (bambi.models.Model): The Bambi GLM model of interest
            posterior (arviz.InferenceData): The posterior arViz object of the
                fitting Bambi model
        """
        self.model = model
        self.family = self.model.family.name
        self.inv_link = self.model.family.link.linkinv
        self.full_params = [
            param for param in self.model.term_names if param in self.model.data.columns
        ]
        self.posterior = posterior
        self.preds = self.model.predict(idata=self.posterior, inplace=False)
        # convert model dataframe to torch design matrix with intercept
        self.n, self.m = self.model.data["y"].shape[0], len(self.model.term_names)
        self.s = self.posterior.posterior.Intercept.values.ravel().shape[0]
        self.y = torch.from_numpy(self.model.data["y"].values).float().to(device)
        self.X_ref = (
            torch.from_numpy(self.model.data[self.full_params].values)
            .float()
            .to(device)
        )
        self.X_ref = torch.concat((self.X_ref, torch.ones((self.n, 1))), dim=1)
        # define optimsation parameters
        self.device = device
        self.n_iters = n_iters
        self.lr = lr

    def project(self, params=None):
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using a PyTorch
        neural network architecture for efficiency.

        Args:
            params (list): The names parameters to use in the restricted model

        Returns:
            torch.tensor: Restricted projection of the reference parameters
        """

        if not params:
            params = self.full_params
        # build restricted data space with intercept in design matrix
        self.X_perp = (
            torch.from_numpy(self.model.data[params].values).float().to(self.device)
        )
        self.X_perp = torch.concat((self.X_perp, torch.ones((self.n, 1))), dim=1)
        # build submodel object
        sub_model = SubModel(self.inv_link, self.s, self.n, self.m)
        sub_model.to(self.device)
        sub_model.zero_grad()
        opt = torch.optim.Adam(sub_model.parameters(), lr=self.lr)
        criterion = KLDivSurrogateLoss(self.family)
        # extract reference model posterior predictions
        y_ast = (
            torch.from_numpy(self.preds.posterior.y_mean.values)
            .float()
            .to(self.device)
            .reshape(-1, self.n)
        )
        # run optimisation loop
        for _ in range(self.n_iters):
            opt.zero_grad()
            y_perp = sub_model(self.X_perp).T
            loss = criterion(y_ast, y_perp)
            loss.backward()
            opt.step()
        # extract projected parameters
        self.theta_perp = list(sub_model.parameters())[0].data
        return self.theta_perp

    def plot_projection(self, params):
        """Plot Kullback-Leibler projection onto a parameter subset.

        Args:
            params (list): The names parameters to use in the restricted model
        """

        if self.theta_perp is None:
            self.theta_perp = self.project(params)
        datadict = {
            "Intercept": self.theta_perp[:, 0],
        }
        paramdict = {
            f"{params[i]}_perp": self.theta_perp[:, i + 1] for i in range(len(params))
        }
        datadict.update(paramdict)
        dataset = az.convert_to_inference_data(datadict)
        az.plot_posterior(dataset)
        plt.show()
