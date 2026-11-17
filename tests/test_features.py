import pandas as pd

from churn_survival import features


def test_first_period_sees_only_the_signup_transaction(handmade_config):
    from churn_survival import panel

    panel.run(handmade_config, verbose=False)
    features.run(handmade_config, verbose=False)
    frame = pd.read_parquet(handmade_config.paths.processed_dir / "features.parquet")
    first = frame.loc[frame["period"] == 0]
    assert (first["n_txns"] == 1).all()
    assert (first["days_since_last_txn"] == 1).all()


def test_cancellation_is_not_visible_before_it_happens(handmade_config):
    from churn_survival import panel

    panel.run(handmade_config, verbose=False)
    features.run(handmade_config, verbose=False)
    users = pd.read_parquet(handmade_config.paths.interim_dir / "users.parquet")
    target = int(users.loc[users["msno"] == "B", "user_id"].iloc[0])
    frame = pd.read_parquet(handmade_config.paths.processed_dir / "features.parquet")
    rows = frame.loc[frame["user_id"] == target].sort_values("period")
    assert rows["n_cancels"].iloc[0] == 0
    assert rows["last_is_cancel"].iloc[0] == 0


def test_age_outside_the_configured_range_becomes_missing(small_config):
    frame = pd.read_parquet(small_config.paths.processed_dir / "features.parquet")
    present = frame["age"].dropna()
    assert present.min() >= small_config.features.age_min
    assert present.max() <= small_config.features.age_max
    assert frame["age"].isna().any()


def test_cumulative_features_never_decrease_within_a_user(small_config):
    frame = pd.read_parquet(small_config.paths.processed_dir / "features.parquet")
    frame = frame.sort_values(["user_id", "period"])
    for column in ("n_txns", "cum_paid", "cum_plan_days", "n_cancels"):
        deltas = frame.groupby("user_id")[column].diff().dropna()
        assert (deltas >= -1e-6).all()
