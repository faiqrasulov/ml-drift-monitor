"""Multivariate drift via a domain classifier: can a model tell reference rows from current rows?"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from driftwatch.univariate import normal_sf


@dataclass
class DomainClassifierResult:
    """Held-out ROC-AUC of reference-vs-current classification and what drives it.

    ``auc`` is the mean of the per-fold held-out AUCs; ``fold_auc`` lists them. ``importances``
    is the mean drop in held-out AUC when one column is permuted: how much of the classifier's
    ability to separate the two samples comes from that column.
    """

    auc: float
    p_value: float
    n_reference: int
    n_current: int
    n_splits: int
    fold_auc: list[float] = field(default_factory=list)
    importances: dict[str, float] = field(default_factory=dict)


def encode_frame(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    categorical: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack both frames into a float matrix (categoricals as integer codes, NaN kept).

    Returns ``X``, the domain label ``y`` (0 = reference, 1 = current) and a boolean mask of
    categorical columns. Categories beyond the 250 most frequent are merged into one code.
    """
    both = pd.concat([reference[list(columns)], current[list(columns)]], ignore_index=True)
    X = np.empty((len(both), len(columns)), dtype=np.float64)
    is_cat = np.zeros(len(columns), dtype=bool)
    for j, col in enumerate(columns):
        s = both[col]
        if col in categorical:
            is_cat[j] = True
            missing = s.isna().to_numpy()
            labels = s.astype(object).astype(str)
            keep = labels[~missing].value_counts().index[:250]
            mapping = {cat: float(i) for i, cat in enumerate(keep)}
            codes = pd.to_numeric(labels.map(mapping), errors="coerce").to_numpy(np.float64)
            codes = np.where(np.isnan(codes), float(len(mapping)), codes)
            codes[missing] = np.nan
            X[:, j] = codes
        else:
            X[:, j] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    y = np.r_[np.zeros(len(reference), dtype=int), np.ones(len(current), dtype=int)]
    return X, y, is_cat


def auc_null_p_value(auc: float, n_reference: int, n_current: int) -> float:
    """One-sided p-value of AUC > 0.5 from the Mann-Whitney normal approximation.

    Valid when the scores were produced by a model that never saw these rows' labels.
    """
    se = math.sqrt((n_reference + n_current + 1) / (12.0 * n_reference * n_current))
    return normal_sf((auc - 0.5) / se)


def domain_classifier_test(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    categorical: Sequence[str] = (),
    n_splits: int = 2,
    max_samples: int = 5000,
    importance: bool = True,
    n_repeats: int = 3,
    seed: int = 0,
) -> DomainClassifierResult:
    """Classifier two-sample test: can gradient boosting tell reference rows from current rows?

    Cross-fitting: the stacked data are split into ``n_splits`` stratified folds; for each fold
    a classifier is trained on the other folds and scored on the held-out fold only. Each
    held-out AUC has an exact Mann-Whitney null (about 0.5 when both samples come from the same
    distribution). The fold p-values are combined with Bonferroni, which stays valid even
    though the folds are dependent. Pooling out-of-fold scores into one AUC is *not* valid:
    every fold's labels train the models that score the other folds, which inflates the AUC's
    null variance (see the README). Each sample is capped at ``max_samples`` rows.
    """
    rng = np.random.default_rng(seed)
    if len(reference) > max_samples:
        reference = reference.iloc[np.sort(rng.choice(len(reference), max_samples, False))]
    if len(current) > max_samples:
        current = current.iloc[np.sort(rng.choice(len(current), max_samples, False))]
    n_ref, n_cur = len(reference), len(current)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if min(n_ref, n_cur) < 2 * n_splits:
        raise ValueError(f"need at least {2 * n_splits} rows in each sample")
    X, y, is_cat = encode_frame(reference, current, columns, categorical)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_auc, fold_p = [], []
    drops = np.zeros(len(columns))
    for train, test in cv.split(X, y):
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            early_stopping=False,
            categorical_features=is_cat if is_cat.any() else None,
            random_state=seed,
        )
        clf.fit(X[train], y[train])
        auc = float(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1]))
        fold_auc.append(auc)
        fold_p.append(auc_null_p_value(auc, int((y[test] == 0).sum()), int(y[test].sum())))
        if importance:
            for j in range(len(columns)):
                for _ in range(n_repeats):
                    Xp = X[test].copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    drops[j] += auc - roc_auc_score(y[test], clf.predict_proba(Xp)[:, 1])
    imp: dict[str, float] = {}
    if importance:
        imp = {c: float(d / (n_splits * n_repeats)) for c, d in zip(columns, drops, strict=True)}
    return DomainClassifierResult(
        auc=float(np.mean(fold_auc)),
        p_value=float(min(1.0, n_splits * min(fold_p))),
        n_reference=n_ref,
        n_current=n_cur,
        n_splits=n_splits,
        fold_auc=fold_auc,
        importances=imp,
    )
