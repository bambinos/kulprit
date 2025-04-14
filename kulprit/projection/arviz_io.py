"""Functions to interact with InferenceData objects"""
import warnings
import numpy as np
from sklearn.cluster import KMeans
from arviz import convert_to_dataset, extract, loo


def get_observed_data(idata, response_name):
    """Extract the observed data from the reference model."""
    observed_data = {response_name: idata.observed_data.get(response_name).to_dict().get("data")}
    return convert_to_dataset(observed_data), idata.observed_data.get(response_name).values


def get_pps(idata, response_name, num_samples, num_clusters, rng):
    """Extract posterior predictive samples from the reference model."""
    pps = extract(
        idata,
        group="posterior_predictive",
        var_names=[response_name],
        num_samples=num_samples,
        rng=rng,
    ).values.T
    pps_tuple = [(pps[i], pps[i - 1]) for i in range(len(pps))]

    ppc, weights = _get_clusters(pps, num_clusters, num_samples)

    return pps_tuple, ppc, weights


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

        print(closest_index)
        representatives.append(pps[closest_index])
        weights = len(cluster_indices) / num_samples

    weights /= np.sum(weights)

    return representatives, weights
