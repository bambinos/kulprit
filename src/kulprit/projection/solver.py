"""Optimisation module."""

from typing import Optional, Tuple
from arviz import InferenceData
from bambi import Model
import numpy as np

import xarray as xr

import torch

from kulprit.data.submodel import SubModel
from kulprit.families.family import Family
from kulprit.projection.losses.kld import KullbackLeiblerLoss
from kulprit.projection.pps import PosteriorPredictive


class Solver:
    def __init__(
        self,
        ref_model: Model,
        ref_idata: InferenceData,
        num_thinned_samples: int = 400,
        num_iters: Optional[int] = 400,
        learning_rate: Optional[float] = 0.01,
    ):
        """Initialise solver object.

        Args:
            num_iters (int, optional): The number of iterations to run the
                optimisation for, defaults to 200
            learning_rate (float, optional): The learning rate for the optimisation
                algorithm, defaults to 0.01
        """

        # log reference model object and fitted inference data
        self.ref_model = ref_model
        self.ref_idata = ref_idata

        # build family and link objects
        self.family = Family(ref_model)
        self.link = self.family.link

        print(self.ref_idata.groups())
        # test posterior predictive distribution has been computed for full model
        if "posterior_predictive" not in self.ref_idata.groups():
            self.ref_model.predict(self.ref_idata, kind="pps", inplace=True)

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
        return (
            torch.from_numpy(
                self.ref_idata.posterior_predictive.stack(samples=("chain", "draw"))
                .transpose(*("samples", ...))[self.ref_model.response.name]
                .values
            )
            .float()
            .mean(-1)
        )

    def optimise(self, res_idata: InferenceData) -> Tuple[np.ndarray, float]:
        """Primary optimisation loop.

        TODO:
            * Allow for flexibility in restricted family. Specifically, allow for
                the restricted model to admit a different family observation model
                to the reference model in general. This may require a more
                fundamental internal refactoring

        Args:
            res_idata (arviz.InferenceData): The restricted model's initial idata
                object

        Returns:
            theta_perp (np.ndarray): The optimisation decision variable solutions
            final_loss (float): The final KL divergence between optimised
                restricted posterior and the reference model posterior
        """

        # build optimisation framework
        self.posterior_predictive = PosteriorPredictive(
            ref_model=self.ref_model, res_idata=res_idata
        )
        self.posterior_predictive.zero_grad()
        optim = torch.optim.Adam(
            self.posterior_predictive.parameters(), lr=self.learning_rate
        )
        loss_fn = KullbackLeiblerLoss(self.family)

        # compute reference model summary statistics
        linear_predictor_ref = self.link.link(self.pps_ast)
        disp_ref = self.family.extract_disp(self.ref_idata).flatten()

        # run optimisation loop
        for _ in range(self.num_iters):
            optim.zero_grad()
            linear_predictor, disp = self.posterior_predictive.forward()
            loss = loss_fn.forward(
                linear_predictor, disp, linear_predictor_ref, disp_ref
            )
            loss.backward()
            optim.step()

        # extract projected parameters and final loss function value
        final_loss = loss.item()
        theta_perp = {
            param[0]: param[1].data
            for param in self.posterior_predictive.named_parameters()
        }
        return theta_perp, final_loss

    def build_idata(self, theta_perp: torch.tensor) -> InferenceData:
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

        if "beta_z" in theta_perp:  # pragma: no cover
            raise NotImplementedError("Hierarchical models not yet implemented")

        if self.posterior_predictive.disp_name in theta_perp:  # pragma: no cover
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
        res_idata_ = self.build_idata(theta_perp)

        # build SubModel object and return
        sub_model = SubModel(
            backend=self.ref_model.backend,
            idata=res_idata_,
            kl_div=kl_div,
            size=len([term for term in term_names if term != "Intercept"]),
            term_names=term_names,
        )
        return sub_model
