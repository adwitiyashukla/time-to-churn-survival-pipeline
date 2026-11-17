from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from .features import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .metrics import StepFunction, kaplan_meier

HAZARD_FLOOR = 1e-6
HAZARD_CEIL = 1.0 - 1e-6


class BaselineHazard:
    def __init__(self):
        self.hazard = None

    def fit(self, frame):
        table = frame.groupby("period")["event"].mean()
        self.hazard = table.reindex(range(int(frame["period"].max()) + 1)).ffill().to_numpy()
        return self

    def predict_hazard(self, frame):
        period = frame["period"].to_numpy()
        clipped = np.clip(period, 0, len(self.hazard) - 1)
        return self.hazard[clipped]


class HorizonHazard:
    def __init__(self, name, factory, columns, horizons):
        self.name = name
        self.factory = factory
        self.columns = columns
        self.horizons = horizons
        self.models = []

    def fit(self, frame, targets):
        design = frame[self.columns]
        self.models = []
        for step in range(1, self.horizons + 1):
            valid, label = targets[step]
            estimator = self.factory()
            estimator.fit(design.loc[valid], label[valid])
            self.models.append(estimator)
        return self

    def predict_hazard(self, cohort, batch_rows=400000):
        pieces = []
        for start in range(0, len(cohort), batch_rows):
            block = cohort[self.columns].iloc[start : start + batch_rows]
            columns = [
                np.clip(model.predict_proba(block)[:, 1], HAZARD_FLOOR, HAZARD_CEIL)
                for model in self.models
            ]
            pieces.append(np.column_stack(columns))
        return np.vstack(pieces) if len(pieces) > 1 else pieces[0]

    def predict_survival(self, cohort, batch_rows=400000):
        return survival_from_hazard(self.predict_hazard(cohort, batch_rows=batch_rows))


def horizon_summary(config):
    panel = pd.read_parquet(
        config.paths.processed_dir / "panel.parquet",
        columns=["user_id", "period", "event", "cause"],
    )
    last_period = panel.groupby("user_id")["period"].max()
    events = panel.loc[panel["event"].to_numpy() == 1].set_index("user_id")
    event_period = events["period"].reindex(last_period.index).fillna(-1)
    event_cause = events["cause"].reindex(last_period.index).fillna(0)
    return pd.DataFrame(
        {
            "last_period": last_period.to_numpy().astype(np.int32),
            "event_period": event_period.to_numpy().astype(np.int32),
            "event_cause": event_cause.to_numpy().astype(np.int8),
        },
        index=last_period.index,
    )


def horizon_targets(frame, summary, horizons, cause=None):
    joined = frame[["user_id", "period"]].join(summary, on="user_id")
    period = joined["period"].to_numpy().astype(np.int32)
    last = joined["last_period"].to_numpy().astype(np.int32)
    event_period = joined["event_period"].to_numpy().astype(np.int32)
    event_cause = joined["event_cause"].to_numpy()
    has_event = event_period >= 0
    targets = {}
    for step in range(1, horizons + 1):
        target_period = period + step - 1
        valid = (target_period <= last) & ~(has_event & (target_period > event_period))
        label = has_event & (target_period == event_period)
        if cause is not None:
            label = label & (event_cause == cause)
        targets[step] = (valid, label.astype(np.int8))
    return targets


def _preprocessor(config, columns):
    numeric = [name for name in columns if name in NUMERIC_FEATURES]
    binary = [name for name in columns if name in BINARY_FEATURES]
    categorical = [name for name in columns if name in CATEGORICAL_FEATURES]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("binary", "passthrough", binary),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=None, sparse_output=True),
                categorical,
            ),
            (
                "tenure",
                SplineTransformer(
                    n_knots=config.models.tenure_spline_knots, degree=3, include_bias=False
                ),
                ["period"],
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def logistic_hazard(config, columns, horizons):
    features = columns + ["period"]

    def factory():
        return Pipeline(
            [
                ("prepare", _preprocessor(config, columns)),
                (
                    "model",
                    LogisticRegression(
                        C=config.models.logistic_c,
                        max_iter=config.models.logistic_max_iter,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    return HorizonHazard("logistic_hazard", factory, features, horizons)


def boosted_factory(config, features):
    categorical = [name in CATEGORICAL_FEATURES for name in features]

    def factory():
        return HistGradientBoostingClassifier(
            max_iter=config.models.boosted_max_iter,
            learning_rate=config.models.boosted_learning_rate,
            max_leaf_nodes=config.models.boosted_max_leaf_nodes,
            min_samples_leaf=config.models.boosted_min_samples_leaf,
            l2_regularization=config.models.boosted_l2,
            early_stopping=True,
            n_iter_no_change=config.models.boosted_early_stopping_rounds,
            validation_fraction=0.1,
            categorical_features=categorical,
            random_state=config.run.seed,
        )

    return factory


def boosted_hazard(config, columns, horizons):
    features = columns + ["period"]
    return HorizonHazard("boosted_hazard", boosted_factory(config, features), features, horizons)


def survival_from_hazard(hazard_matrix):
    return np.cumprod(1.0 - hazard_matrix, axis=1)


def horizon_grid(config, extended=False):
    periods = (
        config.split.extended_horizon_periods if extended else config.split.horizon_periods
    )
    return np.arange(1, periods + 1) * config.panel.period_days


def kaplan_meier_curve(durations, events, times):
    grid, survival, _, _ = kaplan_meier(durations, events)
    step = StepFunction(grid, survival)
    return step(times)


def marginal_survival_matrix(durations, events, times, n_rows):
    curve = kaplan_meier_curve(durations, events, times)
    return np.tile(curve, (n_rows, 1))


def cumulative_incidence(cause_hazards, order):
    names = list(cause_hazards)
    total = sum(cause_hazards[name] for name in names)
    survival = np.cumprod(1.0 - np.clip(total, 0.0, HAZARD_CEIL), axis=1)
    previous = np.concatenate([np.ones((survival.shape[0], 1)), survival[:, :-1]], axis=1)
    return {name: np.cumsum(previous * cause_hazards[name], axis=1) for name in names}
