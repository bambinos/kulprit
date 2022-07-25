import numpy as np
import torch
import torch.nn as nn

from kulprit.families.family import Family


class PosteriorPredictive(nn.Module):
    def __init__(self, ref_model, res_idata):
        """Posterior predictive distribution module.

        Args:
            ref_model (bambi.models.Model): Bambi model of the reference model
                used in the procedure, mostly used to retrieve model information
                and the design matrices
            res_idata (arviz.InferenceData): Infererence data object of the unfitted
                _restricted_ model
        """
        super(PosteriorPredictive, self).__init__()

        # log data surrounding reference model
        self.model = ref_model
        self.idata = res_idata
        self.term_names = list(
            set(self.idata.posterior.data_vars.keys()).intersection(
                self.model.term_names
            )
        )
        self.terms = {term: self.model.terms[term] for term in self.term_names}

        # initialise model family for posterior predictive sampling method
        self.family = Family(model=self.model)
        self.disp_name = self.family.disp_name

        # define model dimensions
        self.chain_n = len(self.idata.posterior.coords.get("chain"))
        self.draw_n = len(self.idata.posterior.coords.get("draw"))

        # initialise model offset terms
        self.x_offsets = []

        # initialise parameter lookup dictionaries
        self.beta_x_lookup = None

        # initialise model parameters
        self.beta_x = (
            torch.nn.Parameter(self.init_beta_x)
            if self.init_beta_x is not None
            else None
        )
        self.beta_z = (
            torch.nn.Parameter(self.init_beta_z)
            if self.init_beta_z is not None
            else None
        )
        self.disp = (
            torch.nn.Parameter(self.init_disp) if self.init_disp is not None else None
        )

    @property
    def offset_terms(self):
        """Return dict of all offset effects in model."""
        return {
            k: v
            for (k, v) in self.terms.items()
            if not v.group_specific and v.kind == "offset"
        }

    @property
    def intercept_term(self):
        """Return the intercept term"""
        term = [
            v
            for v in self.model.terms.values()
            if not v.group_specific and v.kind == "intercept"
        ]
        if term:
            return term[0]
        else:
            return None

    @property
    def common_design(self):
        common_design = None
        if self.model._design.common:
            X = self.model._design.common.design_matrix
            slices = self.model._design.common.slices
            common_design = np.hstack(
                [
                    X[:, slices[term]]
                    for term in self.term_names
                    if term not in self.offset_terms  # don't include offset terms
                ]
            )

            # Add offset columns to their own design matrix
            # Remove them from the common design matrix.
            for term in self.offset_terms:
                term_slice = self.model._design.common.slices[term]
                self.x_offsets.append(common_design[:, term_slice])
            return torch.from_numpy(common_design).float()
        else:
            return None

    @property
    def group_design(self):
        group_design = None
        if self.model._design.group is not None:
            group_design = self.model._design.group.design_matrix
            return torch.from_numpy(group_design).float()
        else:
            return None

    @property
    def init_beta_x(self):
        """Initialise model covariate parameter values.

        Much of this script is adapted from Bambi's prediction method.
        """

        if self.common_design is not None:
            beta_x_list = []
            self.beta_x_lookup = {}
            slice_init = 0

            for name in self.term_names:
                term_dims = list(self.model.terms[name].coords)
                term_posterior = self.idata.posterior[name]
                dims = set(term_posterior.coords)

                # 1-dimensional predictors (a single slope or intercept)
                if dims == {"chain", "draw"}:
                    values = term_posterior.stack(samples=("chain", "draw")).values
                    if len(values.shape) == 1:
                        values = values[:, np.newaxis]
                # 2-dimensional predictors (splines or categoricals)
                elif dims == {"chain", "draw"}.union(term_dims):
                    transpose_dims = ["samples"] + term_dims
                    values = (
                        term_posterior.stack(samples=("chain", "draw"))
                        .transpose(*transpose_dims)
                        .values
                    )
                    if len(values.shape) == 1:
                        values = values[:, np.newaxis]
                else:
                    raise ValueError(f"Unexpected dimensions in term {name}")

                beta_x_list.append(values)

                if len(values.shape) == 2:
                    slc = np.s_[:, slice_init : slice_init + values.shape[1]]
                    slice_init += values.shape[1]
                    self.beta_x_lookup.update(
                        {name: {"values": values, "shape": values.shape, "slice": slc}}
                    )
                else:
                    raise NotImplementedError
            # 'beta_x' is of shape:
            # * (chain_n * draw_n, p) for univariate
            # * (chain_n * draw_n, p, response_n) for multivariate
            return torch.from_numpy(np.hstack(beta_x_list)).float()
        else:
            return None

    @property
    def init_beta_z(self):
        """Initialise model group-level parameter values.

        Much of this script is adapted from Bambi's prediction method.
        """

        if self.group_design is not None:
            beta_z_list = []
            term_names = list(self.model.group_specific_terms)

            for name in term_names:
                term_dims = list(self.model.terms[name].coords)
                factor_dims = [c for c in term_dims if c.endswith("__factor_dim")]
                expr_dims = [c for c in term_dims if c.endswith("__expr_dim")]
                term_posterior = self.idata.posterior[name]
                dims = set(term_posterior.dims)

                # Group-specific term: len(dims) < 3 does not exist.
                # 1 dimensional predictors
                if dims == {"chain", "draw"}.union(expr_dims):
                    transpose_dims = ["samples"] + expr_dims
                    values = (
                        term_posterior.stack(samples=("chain", "draw"))
                        .transpose(*transpose_dims)
                        .values
                    )
                # 2 dimensional predictors
                elif dims == {"chain", "draw"}.union(expr_dims + factor_dims):
                    transpose_dims = ["samples", "coefs"]
                    values = (
                        term_posterior.stack(samples=("chain", "draw"))
                        .stack(coefs=tuple(factor_dims + expr_dims))
                        .transpose(*transpose_dims)
                        .values
                    )
                else:
                    raise ValueError(f"Unexpected dimensions in term {name}")

                beta_z_list.append(values)

            # 'beta_z' is of shape:
            # * (chain_n * draw_n, p) for univariate
            # * (chain_n * draw_n, p, response_n) for multivariate models
            return torch.from_numpy(np.hstack(beta_z_list)).float()
        else:
            return None

    @property
    def init_disp(self):
        """Initialise model dispersion parameter."""

        if self.family.has_dispersion_parameters:
            return self.family.extract_disp(idata=self.idata)
        else:
            return None

    def forward(self):
        """Compute posterior predictive distribution given parameters.

        Returns:
            torch.tensor: The posterior predictive distribution samples of
                shape ``(num_samples, num_obs)``
        """

        # initialise linear predictor value
        linear_predictor = 0

        if self.common_design is not None:
            # 'contribution' is of shape:
            # * (chain_n * draw_n, obs_n) for univariate
            # * (chain_n * draw_n, obs_n, response_n) for multivariate
            if len(self.beta_x.shape) == 2:
                contribution = torch.matmul(self.common_design, self.beta_x.T).T
            else:
                contribution = np.zeros(
                    (
                        self.beta_x.shape[0],
                        self.common_design.shape[0],
                        self.beta_x.shape[2],
                    )
                )
                for i in range(contribution.shape[2]):
                    contribution[:, :, i] = torch.matmul(
                        self.common_design, self.beta_x[:, :, i].T
                    ).T

            shape = (self.chain_n, self.draw_n) + contribution.shape[1:]
            contribution = contribution.reshape(shape)
            linear_predictor += contribution

        # If model contains offset, add directly to the linear predictor
        if self.x_offsets:
            linear_predictor += np.column_stack(self.x_offsets).sum(axis=1)[
                np.newaxis, np.newaxis, :
            ]

        if self.group_design is not None:
            # 'contribution' is of shape:
            # * (chain_n * draw_n, obs_n) for univariate
            # * (chain_n * draw_n, obs_n, response_n) for multivariate
            if len(self.beta_z.shape) == 2:
                contribution = torch.matmul(self.group_design, self.beta_z.T).T
            else:
                contribution = np.zeros(
                    (
                        self.beta_z.shape[0],
                        self.group_design.shape[0],
                        self.beta_z.shape[2],
                    )
                )
                for i in range(contribution.shape[2]):
                    contribution[:, :, i] = torch.matmul(
                        self.group_design, self.beta_z[:, :, i].T
                    ).T

            shape = (self.chain_n, self.draw_n) + contribution.shape[1:]
            contribution = contribution.reshape(shape)
            linear_predictor += contribution

        return linear_predictor.mean(-1).flatten(), self.disp.flatten()
