"""Core family class to be accessed"""

from bambi import Model
from kulprit.families.continuous import GaussianFamily
from kulprit.families.link import Link

# extract families built into Bambi
from bambi.defaults.defaults import BUILTIN_FAMILIES


FAMILIES = {
    "gaussian": GaussianFamily,
}


class Family:
    """Representation of a distribution family.

    Attributes:
        model (bambi.models.Model): The reference Bambi model object
    """

    def __init__(self, model: Model):
        self.name = model.family.name
        if self.name not in FAMILIES:
            raise NotImplementedError(
                f"Family '{self.name}' is not supported.",
            )
        self.link = Link(model.family.link.name)
        self.model = model
        self.family = FAMILIES[self.name](self.model, self.link)
        self.has_dispersion_parameters = self.family.has_dispersion_parameters

        # initialise the name of the dispersion parameter
        self.disp_name = None
        if self.has_dispersion_parameters:
            # extract model parameter names
            response_name = model.response.name
            disp_param = list(
                BUILTIN_FAMILIES.get(self.name).get("likelihood").get("args").keys()
            )[0]
            self.disp_name = f"{response_name}_{disp_param}"

    def kl_div(self, linear_predictor, disp, linear_predictor_ref, disp_ref):
        return self.family.kl_div(linear_predictor, disp, linear_predictor_ref, disp_ref)

    def posterior_predictive(self, **kwargs):
        return self.family.posterior_predictive(**kwargs)

    def extract_disp(self, idata):
        return self.family.extract_disp(idata)
