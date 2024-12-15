"""Functions to interact with InferenceData objects"""
import warnings
from arviz import convert_to_dataset, extract, loo


def get_observed_data(idata, response_name):
    """Extract the observed data from the reference model."""
    observed_data = {response_name: idata.observed_data.get(response_name).to_dict().get("data")}
    return convert_to_dataset(observed_data), idata.observed_data.get(response_name).values


def compute_pps(model, idata):
    """Compute posterior predictive samples from the reference model."""
    if "posterior_predictive" not in idata.groups():
        model.predict(idata, kind="response", inplace=True)


def get_pps(idata, response_name, num_samples):
    """Extract samples posterior predictive samples from the reference model."""
    pps = extract(
        idata,
        group="posterior_predictive",
        var_names=[response_name],
        num_samples=num_samples,
        rng=1,
    ).values.T
    return pps


def compute_loo(submodel=None, idata=None):
    """Compute the PSIS-LOO-CV for a submodel or InferenceData object."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Estimated shape parameter of Pareto distribution"
        )
        if submodel is not None:
            elpd = loo(submodel.idata)
            submodel.elpd_loo = elpd.elpd_loo
            submodel.elpd_se = elpd.se

        if idata is not None:
            return loo(idata)

    return None
