"""Functions to interact with InferenceData objects"""

import warnings
import numpy as np
from sklearn.cluster import KMeans
from arviz_base import extract, from_dict
from arviz_stats import loo


def get_observed_data(idata, response_name):
    """Extract the observed data from the reference model."""
    return idata.observed_data, idata.observed_data.get(response_name).values


def get_new_datatree(posterior, observed_data, log_likelihood):
    return from_dict(
        {"posterior": posterior, "observed_data": observed_data, "log_likelihood": log_likelihood}
    )


def get_pps(idata, response_name, num_samples, num_clusters, rng):
    """Extract posterior predictive samples from the reference model."""
    if num_clusters is not None:
        if num_clusters <= 0:
            raise ValueError("The number of clusters must be positive.")

        if num_clusters > num_samples:
            warnings.warn(
                "The number of clusters is larger than the number of samples. "
                "Setting the number of cluster to the number of samples."
            )
            num_clusters = num_samples

    # Ideally we should thin the posterior and then use all the samples
    total_num_samples = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    pps = extract(
        idata,
        group="posterior_predictive",
        var_names=[response_name],
        num_samples=total_num_samples,
        random_seed=rng,
    ).values.T

    num_samples = min(num_samples, total_num_samples)

    pps_tuple = [(pps[i], pps[i - 1]) for i in range(0, num_samples)]

    if num_clusters is None:
        ppc = pps[:num_samples]
        weights = 1
    else:
        ppc, weights = _get_clusters(pps, num_clusters, total_num_samples)

    return pps_tuple, ppc, weights


def compute_loo(submodel=None, idata=None):
    """Compute the PSIS-LOO-CV for a submodel or InferenceData object."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Estimated shape parameter of Pareto distribution"
        )
        if submodel is not None:
            elpd = loo(submodel.idata)
            submodel.elpd = elpd.elpd
            submodel.elpd_se = elpd.se

        if idata is not None:
            return loo(idata)

    return None


def _get_clusters(pps, num_clusters, num_samples):
    """Get clusters of posterior predictive samples."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pps)
    labels = kmeans.labels_

    representatives = []
    weights = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        cluster_points = pps[labels == cluster_id]
        cluster_indices = np.where(labels == cluster_id)[0]

        centroid = kmeans.cluster_centers_[cluster_id]
        closest_index = cluster_indices[
            np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))
        ]
        representatives.append(pps[closest_index])
        weights = len(cluster_indices) / num_samples

    weights /= np.sum(weights)

    return representatives, weights
