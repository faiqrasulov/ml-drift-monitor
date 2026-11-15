"""California Housing: download with caching and the South/North region split used in the demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES: tuple[str, ...] = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
)
TARGET = "MedHouseVal"
PREDICTION = "prediction"
LATITUDE_SPLIT = 35.0  # south of 35 N = reference region (Los Angeles, San Diego, Inland Empire)


def load_california(data_home: str | Path = "data") -> pd.DataFrame:
    """California Housing (20,640 block groups, 1990 census), cached in ``data_home``.

    The first call downloads about 0.4 MB from the scikit-learn mirror; later calls read the
    cache. Raises ``RuntimeError`` with a clear message when offline and nothing is cached.
    """
    from sklearn.datasets import fetch_california_housing

    Path(data_home).mkdir(parents=True, exist_ok=True)
    try:
        bunch = fetch_california_housing(data_home=str(data_home), as_frame=True)
    except Exception as exc:  # network errors surface as several exception types
        raise RuntimeError(
            "Could not load California Housing. The first run needs internet access; "
            f"afterwards the copy cached in '{data_home}/' is used. Cause: {exc!r}"
        ) from exc
    return bunch.frame.astype("float64")


@dataclass
class RegionSplits:
    """Southern California split three ways, plus the Northern/Central region."""

    train: pd.DataFrame  # fits the price model
    reference: pd.DataFrame  # drift reference and out-of-sample performance baseline
    holdout: pd.DataFrame  # same region, never seen: no-drift windows and synthetic scenarios
    north: pd.DataFrame  # latitude >= 35 N: a real geographic shift


def split_regions(
    df: pd.DataFrame, seed: int = 0, train_frac: float = 0.5, reference_frac: float = 0.25
) -> RegionSplits:
    """Split by latitude, then shuffle the southern rows into train / reference / holdout."""
    south = df[df["Latitude"] < LATITUDE_SPLIT]
    north = df[df["Latitude"] >= LATITUDE_SPLIT]
    idx = np.random.default_rng(seed).permutation(len(south))
    n_train = int(round(train_frac * len(south)))
    n_ref = int(round(reference_frac * len(south)))
    pick = south.iloc
    return RegionSplits(
        train=pick[idx[:n_train]].reset_index(drop=True),
        reference=pick[idx[n_train : n_train + n_ref]].reset_index(drop=True),
        holdout=pick[idx[n_train + n_ref :]].reset_index(drop=True),
        north=north.reset_index(drop=True),
    )
