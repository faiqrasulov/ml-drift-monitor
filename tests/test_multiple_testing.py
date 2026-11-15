import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from driftwatch.multiple_testing import adjust, benjamini_hochberg, bonferroni, reject


@pytest.mark.parametrize("seed", range(8))
def test_bh_matches_statsmodels(seed):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(size=15), rng.uniform(0, 0.01, size=5)])
    p[3] = p[4]  # a tie
    rej, adj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(benjamini_hochberg(p), adj, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(reject(p, 0.05, "bh"), rej)


def test_bonferroni_matches_statsmodels():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    rej, adj, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
    np.testing.assert_allclose(bonferroni(p), adj)
    np.testing.assert_array_equal(reject(p, 0.05, "bonferroni"), rej)


def test_adjust_edge_cases():
    assert adjust([], "bh").size == 0
    np.testing.assert_allclose(adjust([0.2, 0.01], "none"), [0.2, 0.01])
    with pytest.raises(ValueError):
        adjust([0.1], "holm")
