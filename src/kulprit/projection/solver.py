"""Optimisation module."""

from typing import Optional
from arviz import InferenceData
from bambi import Model
import numpy as np

import xarray as xr

import torch

from kulprit.data.submodel import SubModel
from kulprit.projection.losses.kld import KullbackLeiblerLoss
from kulprit.projection.posterior_pred import PosteriorPredictive


class Solver:
    def __init__(
        self,
        ref_model: Model,
        ref_idata: InferenceData,
        num_thinned_samples: int = 400,
        num_iters: Optional[int] = 1_000,
        learning_rate: Optional[float] = 0.01,
    ):
        """Initialise solver object.

        Args:
            data (ModelData): The data object containing the model data of
                the reference model
            family (Family): The family object of the reference model
            num_iters (int, optional): The number of iterations to run the
                optimisation for, defaults to 200
            learning_rate (float, optional): The learning rate for the optimisation
                algorithm, defaults to 0.01
        """

        # log reference model data and family
        self.ref_model = ref_model
        self.ref_idata = ref_idata

        # test posterior predictive distribution has been computed for full model
        if "posterior_predictive" not in self.ref_idata.groups():
            try:
                # make posterior predictive distribution of full model
                self.ref_model.predict(self.ref_idata, kind="pps")
            except Exception as e:
                raise UserWarning(
                    "Please make posterior predictions with the reference ",
                    "model. For more information, kindly consult https://bambin",
                    "os.github.io/bambi/main/api_reference.html#bambi.models.M",
                    "odel.predict",
                ) from e

        # log dimensions of optimisation
        self.num_chain = len(self.ref_idata.posterior_predictive.coords.get("chain"))
        self.num_draw = len(self.ref_idata.posterior_predictive.coords.get("draw"))
        self.num_samples = self.num_chain * self.num_draw
        self.num_thinned_samples = num_thinned_samples
        self.thinned_idx = np.random.randint(
            0, self.num_samples, self.num_thinned_samples
        )

        # log gradient descent parameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate

    @property
    def pps_ast(self):
        """Compute the reference model's posterior predictive distribution."""

        # produce thinned pps
        return torch.from_numpy(
            self.ref_idata.posterior_predictive.stack(samples=("chain", "draw"))
            .transpose(*("samples", ...))[self.ref_model.response.name]
            .values
        ).float()[self.thinned_idx]

    def optimise(self, res_idata):
        """Primary optimisation loop."""

        # build optimisation framework
        self.posterior_predictive = PosteriorPredictive(
            ref_model=self.ref_model, res_idata=res_idata
        )
        self.posterior_predictive.zero_grad()
        optim = torch.optim.Adam(
            self.posterior_predictive.parameters(), lr=self.learning_rate
        )
        loss_fn = KullbackLeiblerLoss()

        # run optimisation loop
        for _ in range(self.num_iters):
            optim.zero_grad()
            pps_perp = self.posterior_predictive.forward()
            loss = loss_fn.forward(self.pps_ast, pps_perp)
            loss.backward()
            optim.step()

        # extract projected parameters and final loss function value
        final_loss = loss.item()
        theta_perp = {
            param[0]: param[1].data
            for param in self.posterior_predictive.named_parameters()
        }
        return theta_perp, final_loss

    def build_idata(self, theta_perp):
        """Build a new restricted idata object given projected posterior."""

        # compute new coordinates
        new_dims = set()
        for term in self.posterior_predictive.term_names:
            new_dims = new_dims.union(
                set(self.posterior_predictive.idata.posterior[term].dims)
            )
        new_coords = {
            dim: np.arange(
                stop=len(
                    self.posterior_predictive.idata.posterior[dim]
                    .coords[dim]
                    .indexes.get(dim)
                ),
                step=1,
            )
            for dim in set(new_dims)
        }

        # initialise new data variables dictionary
        new_data_vars = {}

        if "beta_x" in theta_perp:
            # extract new data variables
            new_data_vars.update(
                {
                    term: (
                        list(self.posterior_predictive.idata.posterior[term].dims),
                        theta_perp["beta_x"][
                            self.posterior_predictive.beta_x_lookup.get(term).get(
                                "slice"
                            )
                        ].reshape(self.posterior_predictive.idata.posterior[term].shape),
                    )
                    for term in self.posterior_predictive.term_names
                }
            )

        if "beta_z" in theta_perp:
            raise NotImplementedError

        if self.posterior_predictive.disp_name in theta_perp:
            new_data_vars.update(
                {
                    self.posterior_predictive.disp_name: (
                        list(
                            self.posterior_predictive.idata.posterior[
                                self.posterior_predictive.disp_name
                            ].dims
                        ),
                        theta_perp["disp"].reshape(
                            self.posterior_predictive.idata.posterior[
                                self.posterior_predictive.disp_name
                            ].shape
                        ),
                    )
                }
            )

        # define submodel attributes
        new_attrs = {"size": len(self.posterior_predictive.term_names)}

        # build restricted posterior object and replace old one
        res_posterior = xr.Dataset(
            data_vars=new_data_vars, coords=new_coords, attrs=new_attrs
        )
        self.res_idata = self.posterior_predictive.idata
        self.res_idata.posterior = res_posterior
        return self.res_idata

    def solve(self, res_idata: InferenceData, term_names: list) -> tuple:
        """Perform projection by gradient descent."""

        # project parameter samples and compute distance from reference model
        theta_perp, kl_div = self.optimise(res_idata)
        res_idata = self.build_idata(theta_perp)

        # build SubModel object and return
        sub_model = SubModel(
            idata=res_idata,
            kl_div=kl_div,
            size=res_idata.posterior.attrs.get("size"),  # TODO: fix size definition
            term_names=term_names,
        )
        return sub_model
