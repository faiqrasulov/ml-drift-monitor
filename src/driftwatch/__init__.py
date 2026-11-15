"""driftwatch: data, prediction and performance drift monitoring with calibrated alarms."""

__version__ = "0.1.0"

from driftwatch.calibration import PSICalibrator, chi2_psi_threshold  # noqa: E402
from driftwatch.multiple_testing import benjamini_hochberg, bonferroni  # noqa: E402
from driftwatch.univariate import (  # noqa: E402
    chi2_homogeneity,
    js_distance,
    ks_test,
    missing_rate_test,
    normalized_wasserstein,
    psi,
    wasserstein_distance,
)

__all__ = [
    "PSICalibrator",
    "benjamini_hochberg",
    "bonferroni",
    "chi2_homogeneity",
    "chi2_psi_threshold",
    "js_distance",
    "ks_test",
    "missing_rate_test",
    "normalized_wasserstein",
    "psi",
    "wasserstein_distance",
]
