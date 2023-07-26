"""Load and pre-process datasets for tutorials."""

import bambi as bmb
import numpy as np
import pandas as pd


def load_adults_data() -> pd.DataFrame:
    """
    Load and pre-process the adults dataset.

    Much of this code is taken from bambi tutorials to simplify the workflow in
    kulprit tutorials. This dataset is used in Bernoulli family models.

    Returns:
        pd.DataFrame: The loaded and pre-processed adults dataset.
    """

    data = bmb.load_data("adults")
    categorical_cols = data.columns[data.dtypes == object].tolist()
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    data = data[data["race"].isin(["Black", "White"])]
    data["race"] = data["race"].cat.remove_unused_categories()

    age_mean = np.mean(data["age"])
    age_std = np.std(data["age"])
    hs_mean = np.mean(data["hs_week"])
    hs_std = np.std(data["hs_week"])

    data["age"] = (data["age"] - age_mean) / age_std
    data["age2"] = data["age"] ** 2
    data["age3"] = data["age"] ** 3
    data["hs_week"] = (data["hs_week"] - hs_mean) / hs_std
    data["hs_week2"] = data["hs_week"] ** 2
    data["hs_week3"] = data["hs_week"] ** 3
    return data


def load_batting_data() -> pd.DataFrame:
    """
    Load and pre-process the batting dataset.

    Much of this code is taken from bambi tutorials to simplify the workflow in
    kulprit tutorials. This dataset is used in binomial family models.

    Returns:
        pd.DataFrame: The loaded and pre-processed batting dataset.
    """

    df = bmb.load_data("batting")
    df["AB"] = df["AB"].replace(0, np.nan)
    df = df.dropna()
    df["batting_avg"] = df["H"] / df["AB"]
    df = df.astype({"H": int, "AB": int})
    df = df[df["yearID"] >= 2016]
    df = df.iloc[0:15]
    return df
