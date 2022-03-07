"""Projection class."""

import arviz as az
import torch

from .submodel import KLDivSurrogateLoss, SubModel


class Projector:
    def __init__(self, model, posterior):
        """Reference model builder for projection predictive model selection.

        This object initialises the reference model and handles the core
        projection and variable search methods of the model selection procedure.

        Args:
            model (bambi.models.Model): The Bambi GLM model of interest
            posterior (arviz.InferenceData): The posterior arViz object of the
                fitting Bambi model
            device (torch.device): The PyTorch device being used
        """
        # define model-specific attributed
        self.model = model
        self.family = self.model.family.name
        self.inv_link = self.model.family.link.linkinv
        self.full_params = [
            param for param in self.model.term_names if param in self.model.data.columns
        ]
        self.posterior = posterior
        self.preds = self.model.predict(idata=self.posterior, inplace=False)
        # convert model dataframe to torch design matrix with intercept
        self.n, self.m = (
            self.model.data["y"].shape[0],
            len(self.model.term_names),
        )
        self.s = self.posterior.posterior.Intercept.values.ravel().shape[0]
        self.y = torch.from_numpy(self.model.data["y"].values).float()
        self.X_ref = torch.from_numpy(self.model.data[self.full_params].values).float()
        self.X_ref = torch.concat((self.X_ref, torch.ones((self.n, 1))), dim=1)

    def project(
        self,
        params=None,
        num_iters=200,
        learning_rate=0.01,
    ):
        """Primary projection method for GLM reference model.

        The projection is defined as the values of the submodel parameters
        minimising the Kullback-Leibler divergence between the submodel
        and the reference model. This is perform numerically using a PyTorch
        neural network architecture for efficiency.

        Todo:
            * Project dispersion parameters if present in reference distribution

        Args:
            params (list): The names parameters to use in the restricted model
            num_iters (int): Number of iterations over which to run backprop
            learning_rate (float): Backprop optimiser's learning rate

        Returns:
            torch.tensor: Restricted projection of the reference parameters
        """

        # test restricted parameter set
        if not params:
            params = self.full_params
        self.p = len(params) + 1

        # define optimisation parameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        # build restricted data space with intercept in design matrix
        self.X_perp = torch.from_numpy(self.model.data[params].values).float()
        self.X_perp = torch.concat((self.X_perp, torch.ones((self.n, 1))), dim=1)
        assert self.X_perp.shape == (self.n, self.p,), (
            f"Expected variates dimensions {(self.n, self.p,)}, "
            + f"received {self.X_perp.shape}."
        )
        # build submodel object
        sub_model = SubModel(self.inv_link, self.s, self.n, self.p)
        sub_model
        sub_model.zero_grad()
        opt = torch.optim.Adam(sub_model.parameters(), lr=self.learning_rate)
        criterion = KLDivSurrogateLoss(self.family)
        # extract reference model posterior predictions
        y_ast = (
            torch.from_numpy(self.preds.posterior.y_mean.values)
            .float()
            .reshape(self.s, self.n)
        )
        # run optimisation loop
        for _ in range(self.num_iters):
            opt.zero_grad()
            y_perp = sub_model(self.X_perp)
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

        if not hasattr(self, "theta_perp"):
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
