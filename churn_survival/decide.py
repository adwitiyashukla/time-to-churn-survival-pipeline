from __future__ import annotations

import numpy as np
import pandas as pd

from . import models

MIN_PLAN_DAYS = 1.0


def _monthly_value(frame, period_days):
    plan_days = np.maximum(frame["last_plan_days"].to_numpy(dtype=np.float64), MIN_PLAN_DAYS)
    paid = np.maximum(frame["last_paid"].to_numpy(dtype=np.float64), 0.0)
    return paid * (period_days / plan_days)


def _policy(name, ranking, frame, config, horizon_days):
    count = max(int(round(len(frame) * config.decide.target_fraction)), 1)
    order = np.argsort(-ranking, kind="stable")[:count]
    selected = frame.iloc[order]
    observed_churn = (
        (selected["event"].to_numpy() == 1) & (selected["duration"].to_numpy() <= horizon_days)
    ).astype(np.float64)
    value = selected["residual_value"].to_numpy()
    saved = config.decide.save_rate * float(np.sum(observed_churn * value))
    expected = config.decide.save_rate * float(
        np.sum(selected["churn_probability"].to_numpy() * value)
    )
    cost = config.decide.contact_cost * count
    return {
        "policy": name,
        "targeted": count,
        "churners_caught": int(observed_churn.sum()),
        "precision": float(observed_churn.mean()),
        "expected_value_saved": expected,
        "realised_value_saved": saved,
        "contact_cost": cost,
        "realised_net_gain": saved - cost,
    }


def run(config, verbose=True):
    path = config.paths.artifacts_dir / "test_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError("run the evaluate stage before decide, no file at %s" % path)
    frame = pd.read_parquet(path)
    times = models.horizon_grid(config)
    extended = models.horizon_grid(config, extended=True)
    columns = ["survival_%d" % int(time) for time in times]
    survival = frame[columns].to_numpy(dtype=np.float64)

    period_days = config.panel.period_days
    monthly = _monthly_value(frame, period_days)
    expected_periods = survival.sum(axis=1)
    tail = survival[:, -1]
    remaining = len(extended) - len(times)
    if remaining > 0:
        ratio = np.clip(survival[:, -1] / np.clip(survival[:, -2], 1e-9, None), 0.0, 1.0)
        extra = tail * ratio * (1.0 - ratio**remaining) / np.clip(1.0 - ratio, 1e-9, None)
        expected_periods = expected_periods + extra

    frame["monthly_value"] = monthly
    frame["expected_periods"] = expected_periods
    frame["residual_value"] = monthly * expected_periods * config.decide.period_margin_rate
    frame["churn_probability"] = 1.0 - survival[:, -1]
    frame["value_at_risk"] = frame["churn_probability"] * frame["residual_value"]

    horizon_days = float(times[-1])
    rng = np.random.default_rng(config.run.seed)
    policies = [
        _policy("churn_probability", frame["churn_probability"].to_numpy(), frame, config, horizon_days),
        _policy("value_at_risk", frame["value_at_risk"].to_numpy(), frame, config, horizon_days),
        _policy("random", rng.random(len(frame)), frame, config, horizon_days),
    ]
    summary = pd.DataFrame(policies)
    summary.to_csv(config.paths.artifacts_dir / "targeting.csv", index=False)
    frame.to_parquet(config.paths.artifacts_dir / "customer_value.parquet", index=False)
    if verbose:
        print(summary.to_string(index=False))
        print(
            "mean expected remaining periods %.2f  mean residual value %.1f"
            % (frame["expected_periods"].mean(), frame["residual_value"].mean())
        )
    return {"policies": policies}
