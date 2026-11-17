from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import HistGradientBoostingClassifier

from . import metrics, models, splits
from .features import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .panel import CAUSE_PASSIVE, CAUSE_VOLUNTARY

COX_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES


def _apply_allowed(frame, allowed):
    for name, keep in allowed.items():
        values = frame[name].to_numpy()
        frame[name] = np.where(np.isin(values, keep), values, splits.RARE_CODE).astype(np.int16)
    return frame


def _score_model(label, survival, durations, events, times, censor_fn, bins):
    risk = 1.0 - survival[:, -1]
    horizon = float(times[-1])
    rows = metrics.calibration_table(survival[:, -1], durations, events, horizon, bins=bins)
    result = {
        "model": label,
        "uno_c": metrics.uno_c_index(durations, events, risk, horizon, censor_fn),
        "ibs": metrics.integrated_brier_score(times, survival, durations, events, censor_fn),
        "calibration_error": metrics.calibration_error(rows),
    }
    for index, time in enumerate(times):
        result["auc_%d" % int(time)] = metrics.cumulative_dynamic_auc(
            float(time), 1.0 - survival[:, index], durations, events, censor_fn
        )
    return result, rows


def cox_fit_transform(cohort, columns):
    frame = cohort[columns].astype(np.float32)
    median = frame.median()
    frame = frame.fillna(median)
    spread = frame.std()
    keep = [name for name in columns if spread[name] > 1e-8]
    frame = frame[keep]
    centre = frame.mean()
    scale = frame.std()
    transform = {"keep": keep, "median": median, "centre": centre, "scale": scale}
    return (frame - centre) / scale, transform


def cox_apply_transform(cohort, transform):
    keep = transform["keep"]
    frame = cohort[keep].astype(np.float32)
    frame = frame.fillna(transform["median"][keep])
    return (frame - transform["centre"]) / transform["scale"]


def _cox_survival_batched(fitter, cohort, transform, times, batch_rows=400000):
    pieces = []
    for start in range(0, len(cohort), batch_rows):
        block = cohort.iloc[start : start + batch_rows]
        design = cox_apply_transform(block, transform)
        pieces.append(fitter.predict_survival_function(design, times=times).to_numpy().T)
        del design
    return np.vstack(pieces) if len(pieces) > 1 else pieces[0]


def _fit_cox(config, cohort, rng):
    sample = cohort
    if len(cohort) > config.models.cox_sample_users:
        index = rng.choice(len(cohort), size=config.models.cox_sample_users, replace=False)
        sample = cohort.iloc[index]
    design, transform = cox_fit_transform(sample, COX_COLUMNS)
    design["duration"] = np.maximum(sample["duration"].to_numpy(), 1)
    design["event"] = sample["event"].to_numpy()
    fitter = CoxPHFitter(penalizer=config.models.cox_penalizer)
    fitter.fit(design, duration_col="duration", event_col="event")
    return fitter, transform, design


def _naive_labels(config, frame, spells):
    horizon_days = config.split.horizon_periods * config.panel.period_days
    confirmable_until = splits.to_day(config.split.train_end) - config.panel.grace_days
    lookup = spells.set_index("user_id")[["end_day", "event"]]
    joined = frame[["user_id", "period_start_day"]].join(lookup, on="user_id")
    end_day = joined["end_day"].to_numpy()
    observed = joined["event"].to_numpy().astype(bool)
    start = joined["period_start_day"].to_numpy()
    within = (end_day - start) <= horizon_days
    confirmed = end_day <= confirmable_until
    return within & observed & confirmed


def _naive_baseline(config, train, test, columns, times):
    labels = train["naive_label"].to_numpy().astype(np.int8)
    categorical = [name in CATEGORICAL_FEATURES for name in columns]
    estimator = HistGradientBoostingClassifier(
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
    estimator.fit(train[columns], labels)
    probability = estimator.predict_proba(test[columns])[:, 1]
    final = np.clip(1.0 - probability, 1e-6, 1 - 1e-6)
    survival = np.tile(final.reshape(-1, 1), (1, len(times)))
    for index in range(len(times) - 1):
        share = (index + 1) / len(times)
        survival[:, index] = 1.0 - (1.0 - final) * share
    return survival, probability


def run(config, verbose=True):
    rng = np.random.default_rng(config.run.seed)
    config.paths.ensure()
    spells = pd.read_parquet(config.paths.processed_dir / "spells.parquet")

    train = splits.load_training(config)
    allowed = splits.fit_rare(config, train)
    train = splits.apply_rare(train, allowed)
    train["naive_label"] = _naive_labels(config, train, spells)

    columns = list(FEATURE_COLUMNS)
    horizons = config.split.horizon_periods
    summary = models.horizon_summary(config)
    targets = models.horizon_targets(train, summary, horizons)
    fitted = {}
    if config.models.fit_logistic_hazard:
        fitted["logistic_hazard"] = models.logistic_hazard(config, columns, horizons).fit(
            train, targets
        )
    if config.models.fit_boosted_hazard:
        fitted["boosted_hazard"] = models.boosted_hazard(config, columns, horizons).fit(
            train, targets
        )

    test = splits.landmark_cohort(
        config,
        splits.load_landmark_window(config, config.split.test_landmark),
        spells,
        config.split.test_landmark,
    )
    test = _apply_allowed(test, allowed)
    train_landmark = splits.landmark_cohort(
        config,
        splits.load_landmark_window(config, config.split.train_end),
        spells,
        config.split.train_end,
    )
    train_landmark = _apply_allowed(train_landmark, allowed)

    durations = test["duration"].to_numpy().astype(np.float64)
    events = test["event"].to_numpy().astype(bool)
    censor_times, censor_values = metrics.censoring_curve(durations, events)
    censor_fn = metrics.StepFunction(censor_times, censor_values)
    times = models.horizon_grid(config).astype(np.float64)

    results = []
    calibrations = {}
    curves = {}

    if config.models.fit_kaplan_meier:
        baseline = models.marginal_survival_matrix(
            train_landmark["duration"].to_numpy(),
            train_landmark["event"].to_numpy(),
            times,
            len(test),
        )
        curves["kaplan_meier"] = baseline

    for label, model in fitted.items():
        curves[label] = model.predict_survival(test)

    if config.models.fit_cox:
        fitter, transform, design = _fit_cox(config, train_landmark, rng)
        curves["cox_ph"] = _cox_survival_batched(fitter, test, transform, times)
        check = proportional_hazard_test(fitter, design, time_transform="rank")
        ph = check.summary["p"].sort_values()
        (config.paths.artifacts_dir / "schoenfeld.csv").write_text(
            ph.to_frame("p_value").to_csv()
        )

    if config.evaluate.naive_baseline:
        naive_survival, _ = _naive_baseline(config, train, test, columns, times)
        curves["naive_fixed_window"] = naive_survival

    for label, survival in curves.items():
        result, rows = _score_model(
            label, survival, durations, events, times, censor_fn, config.evaluate.calibration_bins
        )
        results.append(result)
        calibrations[label] = rows

    table = pd.DataFrame(results).sort_values("ibs", ascending=True).reset_index(drop=True)
    table.to_csv(config.paths.artifacts_dir / "model_metrics.csv", index=False)

    calibration_rows = []
    for label, rows in calibrations.items():
        for row in rows:
            calibration_rows.append({"model": label, **row})
    pd.DataFrame(calibration_rows).to_csv(
        config.paths.artifacts_dir / "calibration.csv", index=False
    )

    best = table.iloc[0]["model"]
    predictions = test[["user_id", "duration", "event", "cause", "last_paid", "last_plan_days"]].copy()
    for index, time in enumerate(times):
        predictions["survival_%d" % int(time)] = curves[best][:, index]
    predictions.to_parquet(
        config.paths.artifacts_dir / "test_predictions.parquet", index=False
    )

    cause_summary = {}
    if config.models.fit_competing_risks:
        cause_hazard = {}
        builder = (
            models.boosted_hazard
            if config.models.competing_risks_model == "boosted"
            else models.logistic_hazard
        )
        for label, cause in (("voluntary", CAUSE_VOLUNTARY), ("passive", CAUSE_PASSIVE)):
            cause_targets = models.horizon_targets(train, summary, horizons, cause=cause)
            model = builder(config, columns, horizons).fit(train, cause_targets)
            cause_hazard[label] = model.predict_hazard(test)
        incidence = models.cumulative_incidence(cause_hazard, list(cause_hazard))
        for label, matrix in incidence.items():
            cause_summary[label] = float(matrix[:, horizons - 1].mean())
        cause_summary["observed_voluntary"] = float(
            ((test["cause"].to_numpy() == CAUSE_VOLUNTARY) & (durations <= times[-1])).mean()
        )
        cause_summary["observed_passive"] = float(
            ((test["cause"].to_numpy() == CAUSE_PASSIVE) & (durations <= times[-1])).mean()
        )
        cause_summary["observed_total"] = float((events & (durations <= times[-1])).mean())
        cause_summary["predicted_total"] = float(
            cause_summary["voluntary"] + cause_summary["passive"]
        )
        pd.Series(cause_summary).to_csv(
            config.paths.artifacts_dir / "competing_risks.csv", header=False
        )

    report = {
        "landmark": str(config.split.test_landmark),
        "cohort_size": int(len(test)),
        "events_in_window": int(events.sum()),
        "censored_before_horizon": int(((~events) & (durations < times[-1])).sum()),
        "best_model": best,
        "metrics": results,
        "competing_risks": cause_summary,
    }
    (config.paths.artifacts_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(table.to_string(index=False))
        if cause_summary:
            print(cause_summary)
    return report
