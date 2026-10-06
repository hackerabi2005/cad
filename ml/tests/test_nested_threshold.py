import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from ml.train import nested_threshold_cv  # returns the outer-fold sensitivities


def test_nested_threshold_survives_an_overfitting_model():
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=3,
        flip_y=0.25,
        weights=[0.6, 0.4],
        random_state=0,
    )
    # a deep forest memorises its training fold; tuning on in-sample
    # predictions would pick a very high cutoff and wreck outer sensitivity
    make_model = lambda: RandomForestClassifier(n_estimators=40, random_state=0)
    sens = nested_threshold_cv(make_model, X, y, target_sensitivity=0.90, seed=0)
    assert np.mean(sens) >= 0.80
