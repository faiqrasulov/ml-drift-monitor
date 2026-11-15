"""driftwatch: data, prediction and performance drift monitoring with calibrated alarms."""

__version__ = "0.1.0"

from driftwatch.calibration import PSICalibrator, chi2_psi_threshold  # noqa: E402
from driftwatch.monitor import (  # noqa: E402
    ColumnDrift,
    DriftConfig,
    DriftMonitor,
    DriftReport,
    compare,
)
from driftwatch.multiple_testing import benjamini_hochberg, bonferroni  # noqa: E402
from driftwatch.multivariate import domain_classifier_test  # noqa: E402
from driftwatch.performance import performance_check, rolling_metrics  # noqa: E402
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
    "ColumnDrift",
    "DriftConfig",
    "DriftMonitor",
    "DriftReport",
    "PSICalibrator",
    "benjamini_hochberg",
    "bonferroni",
    "chi2_homogeneity",
    "chi2_psi_threshold",
    "compare",
    "domain_classifier_test",
    "js_distance",
    "ks_test",
    "missing_rate_test",
    "normalized_wasserstein",
    "performance_check",
    "psi",
    "rolling_metrics",
    "wasserstein_distance",
]
