import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from spectralselector import VIPSelector

chemotools = pytest.importorskip("chemotools")
StandardNormalVariate = pytest.importorskip("chemotools.scatter").StandardNormalVariate


def test_chemotools_ndarray_output_flows_into_selector():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(50, 14)) + np.linspace(1, 2, 14)
    y = 2 * X[:, 3] - X[:, 10] + rng.normal(scale=0.1, size=50)
    pipeline = Pipeline(
        [
            ("snv", StandardNormalVariate()),
            ("select", VIPSelector(n_components=2, n_features_to_select=6)),
            ("model", PLSRegression(n_components=2)),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=4),
        scoring="neg_root_mean_squared_error",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
