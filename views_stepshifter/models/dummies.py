"""Feature-blind hurdle-stage estimators for stage-focused hyperparameter sweeps.

A DshurfModel sweep normally searches the joint clf/reg space. Pinning one
stage to a dummy estimator here narrows the sweep to the other stage: the
dummy contributes no hyperparameters, no covariates, and (per step-fit) a
constant signal, so trial-to-trial metric differences are attributable to
the real stage alone.

The dummies double as baselines: dummy clf + real reg is the "reg-only"
model, real clf + dummy reg is the climatology-magnitude model, and both
dummies together form the null model.

- ``PriorDummyClassifier``: sklearn ``DummyClassifier`` locked to
  ``strategy="prior"`` — ``predict_proba(X)[:, 1]`` returns the training
  base rate ``P(y > 0)`` for every row.
- ``EmpiricalDummyRegressor``: stores the positive training values at fit
  time; ``predict`` returns, per row, ``sample_size`` draws with
  replacement from that empirical distribution (the best feature-blind
  guess of the positive-magnitude distribution).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted


class PriorDummyClassifier(DummyClassifier):
    """DummyClassifier locked to ``strategy="prior"``.

    ``fit`` records the training class frequencies; ``predict_proba``
    returns ``[1 - p, p]`` for every row, where ``p`` is the training
    share of positive labels. Constructed with no arguments by the hurdle
    pipelines (empty parameter dicts), hence the explicit no-arg
    ``__init__`` that fixes the strategy.
    """

    def __init__(self):
        super().__init__(strategy="prior")


class EmpiricalDummyRegressor(BaseEstimator):
    """Feature-blind distributional regressor for the hurdle positive stage.

    ``fit`` stores the positive training values; ``predict`` returns, for
    each row, ``sample_size`` values drawn with replacement from that
    empirical distribution — the "climatology" magnitude forecast. It
    exposes no hyperparameters (empty parameter dicts are valid).

    Note: requires at least one positive training value; the hurdle fit
    loop guarantees this before the positive stage is fit.
    """

    def __init__(self):
        # Explicit no-arg __init__ so `Estimator(**{})` from empty clf/reg
        # parameter dicts constructs cleanly.
        pass

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("EmpiricalDummyRegressor fit on an empty label set.")
        self.values_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X, sample_size=1000):
        """Bootstrap ``sample_size`` draws per row from the fitted values."""
        check_is_fitted(self, "is_fitted_")
        n_rows = len(X)
        rng = np.random.default_rng()
        return rng.choice(self.values_, size=(n_rows, sample_size), replace=True)


def dummy_predict(model: EmpiricalDummyRegressor, X, sample_size: int) -> np.ndarray:
    """PREDICTION_WRAPPERS adapter: (len(X), sample_size) bootstrap draws."""
    return model.predict(X, sample_size=sample_size)
