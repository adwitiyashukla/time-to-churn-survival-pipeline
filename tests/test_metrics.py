import numpy as np

from churn_survival import metrics


def test_kaplan_meier_matches_a_hand_computed_curve():
    durations = np.array([1, 2, 3, 4, 5])
    events = np.array([1, 1, 0, 1, 1])
    times, survival, _, _ = metrics.kaplan_meier(durations, events)
    np.testing.assert_allclose(times, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(survival, [0.8, 0.6, 0.6, 0.3, 0.0])


def test_censoring_curve_is_flat_when_nothing_is_censored():
    durations = np.array([1, 2, 3, 4])
    events = np.ones(4, dtype=bool)
    _, survival = metrics.censoring_curve(durations, events)
    np.testing.assert_allclose(survival, np.ones(4))


def test_perfect_ranking_gives_a_concordance_of_one():
    durations = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    events = np.ones(5, dtype=bool)
    censor = metrics.StepFunction(np.array([0.0]), np.array([1.0]))
    score = metrics.uno_c_index(durations, events, -durations, 100.0, censor)
    assert score == 1.0
    reversed_score = metrics.uno_c_index(durations, events, durations, 100.0, censor)
    assert reversed_score == 0.0


def test_brier_score_is_zero_for_a_perfect_forecast():
    durations = np.array([10.0, 20.0, 30.0, 40.0])
    events = np.ones(4, dtype=bool)
    censor = metrics.StepFunction(np.array([0.0]), np.array([1.0]))
    times = np.array([25.0])
    survival = (durations > 25.0).astype(float).reshape(-1, 1)
    scores = metrics.brier_score(times, survival, durations, events, censor)
    np.testing.assert_allclose(scores, [0.0])


def test_auc_separates_a_perfect_score_from_a_useless_one():
    rng = np.random.default_rng(3)
    n = 500
    risk = rng.normal(size=n)
    durations = 100.0 - 30.0 * risk + rng.normal(scale=1.0, size=n)
    events = np.ones(n, dtype=bool)
    censor = metrics.StepFunction(np.array([0.0]), np.array([1.0]))
    good = metrics.cumulative_dynamic_auc(100.0, risk, durations, events, censor)
    useless = metrics.cumulative_dynamic_auc(100.0, rng.normal(size=n), durations, events, censor)
    assert good > 0.98
    assert 0.4 < useless < 0.6


def test_calibration_error_is_small_when_the_model_is_right():
    rng = np.random.default_rng(11)
    n = 6000
    probability = rng.uniform(0.1, 0.9, size=n)
    churned = rng.random(n) < (1.0 - probability)
    durations = np.where(churned, 10.0, 100.0)
    events = churned
    rows = metrics.calibration_table(probability, durations, events, 50.0, bins=8)
    assert metrics.calibration_error(rows) < 0.03


def test_cox_transform_is_fitted_once_and_reused(small_config):
    import pandas as pd

    from churn_survival import evaluate, splits

    spells = pd.read_parquet(small_config.paths.processed_dir / "spells.parquet")
    window = splits.load_landmark_window(small_config, small_config.split.train_end)
    cohort = splits.landmark_cohort(
        small_config, window, spells, small_config.split.train_end
    )
    design, transform = evaluate.cox_fit_transform(cohort, evaluate.COX_COLUMNS)

    block = cohort.copy()
    block["last_is_cancel"] = 0
    block["auto_renew_dropped"] = 0
    applied = evaluate.cox_apply_transform(block, transform)
    assert list(applied.columns) == list(design.columns)

    half = evaluate.cox_apply_transform(cohort.iloc[: len(cohort) // 2], transform)
    np.testing.assert_allclose(
        half.to_numpy(), design.to_numpy()[: len(half)], rtol=1e-5, atol=1e-5
    )
