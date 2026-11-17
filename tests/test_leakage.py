import numpy as np
import pandas as pd

from churn_survival import features

TRANSACTION_COLUMNS = [
    "user_id",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "is_cancel",
    "txn_day",
    "expire_day",
]


def _inputs(config):
    transactions = pd.read_parquet(
        config.paths.interim_dir / "transactions.parquet", columns=TRANSACTION_COLUMNS
    )
    panel = pd.read_parquet(config.paths.processed_dir / "panel.parquet")
    panel = panel.sort_values(["user_id", "period"], kind="stable").reset_index(drop=True)
    members = pd.read_parquet(config.paths.interim_dir / "members.parquet").set_index("user_id")
    return transactions, panel, members


def test_features_ignore_everything_at_or_after_the_period_start(small_config):
    transactions, panel, members = _inputs(small_config)
    cutoff = int(np.quantile(panel["period_start_day"], 0.6))
    block = panel.loc[panel["period_start_day"] <= cutoff].reset_index(drop=True)
    assert len(block) > 0

    full = features._compute_block(
        small_config, features.TransactionHistory(transactions), block, members
    )
    truncated = features._compute_block(
        small_config,
        features.TransactionHistory(transactions.loc[transactions["txn_day"] < cutoff]),
        block,
        members,
    )
    pd.testing.assert_frame_equal(full, truncated)


def test_corrupting_the_future_does_not_move_a_single_feature(small_config):
    transactions, panel, members = _inputs(small_config)
    cutoff = int(np.quantile(panel["period_start_day"], 0.5))
    block = panel.loc[panel["period_start_day"] <= cutoff].reset_index(drop=True)

    poisoned = transactions.copy()
    future = poisoned["txn_day"].to_numpy() >= cutoff
    assert future.sum() > 0
    for column in ("payment_method_id", "payment_plan_days", "plan_list_price"):
        poisoned.loc[future, column] = 999
    poisoned.loc[future, "actual_amount_paid"] = 0
    poisoned.loc[future, "is_auto_renew"] = 0
    poisoned.loc[future, "is_cancel"] = 1

    clean = features._compute_block(
        small_config, features.TransactionHistory(transactions), block, members
    )
    dirty = features._compute_block(
        small_config, features.TransactionHistory(poisoned), block, members
    )
    pd.testing.assert_frame_equal(clean, dirty)


def test_brute_force_recomputation_matches_the_vectorised_features(small_config):
    transactions, panel, members = _inputs(small_config)
    computed = features._compute_block(
        small_config, features.TransactionHistory(transactions), panel, members
    )
    rng = np.random.default_rng(0)
    sample = rng.choice(len(panel), size=200, replace=False)
    grouped = {
        user: group.sort_values("txn_day")
        for user, group in transactions.groupby("user_id", sort=False)
    }
    for position in sample:
        row = panel.iloc[int(position)]
        history = grouped[row["user_id"]]
        history = history.loc[history["txn_day"] < row["period_start_day"]]
        expected_count = len(history)
        assert computed["n_txns"].iloc[int(position)] == expected_count
        if expected_count == 0:
            continue
        assert computed["cum_paid"].iloc[int(position)] == history["actual_amount_paid"].sum()
        assert computed["n_cancels"].iloc[int(position)] == history["is_cancel"].sum()
        last = history.iloc[-1]
        assert computed["last_payment_method_id"].iloc[int(position)] == last["payment_method_id"]
        assert (
            computed["days_since_last_txn"].iloc[int(position)]
            == row["period_start_day"] - last["txn_day"]
        )
