from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DAY_SCALE = 100000

NUMERIC_FEATURES = [
    "tenure_days",
    "days_since_last_txn",
    "days_to_expiry",
    "membership_days_left",
    "last_plan_days",
    "last_list_price",
    "last_paid",
    "last_discount_ratio",
    "price_delta",
    "n_txns",
    "n_cancels",
    "auto_renew_share",
    "discount_share",
    "cum_paid",
    "mean_paid",
    "cum_plan_days",
    "n_txns_recent",
    "n_cancels_recent",
    "paid_recent",
    "days_since_registration",
    "age",
]

BINARY_FEATURES = [
    "last_is_auto_renew",
    "last_is_cancel",
    "price_increased",
    "method_changed",
    "plan_changed",
    "auto_renew_dropped",
    "has_member_record",
]

CATEGORICAL_FEATURES = [
    "last_payment_method_id",
    "city",
    "registered_via",
    "gender_code",
    "calendar_month",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
KEY_COLUMNS = ["user_id", "period", "period_start_day", "event", "cause"]
RARE_CODE = 9998
MISSING_CODE = 9999


def _schema():
    fields = [
        ("user_id", pa.int32()),
        ("period", pa.int16()),
        ("period_start_day", pa.int32()),
        ("event", pa.int8()),
        ("cause", pa.int8()),
    ]
    fields += [(name, pa.float32()) for name in NUMERIC_FEATURES]
    fields += [(name, pa.int8()) for name in BINARY_FEATURES]
    fields += [(name, pa.int16()) for name in CATEGORICAL_FEATURES]
    return pa.schema(fields)


def _prefix(values, dtype):
    out = np.zeros(len(values) + 1, dtype=dtype)
    np.cumsum(values, out=out[1:], dtype=dtype)
    return out


HISTORY_COLUMNS = [
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


def load_history(config):
    table = pq.read_table(
        config.paths.interim_dir / "transactions.parquet", columns=HISTORY_COLUMNS
    )
    arrays = {
        name: table.column(name).to_numpy(zero_copy_only=False) for name in HISTORY_COLUMNS
    }
    del table
    return TransactionHistory(arrays)


class TransactionHistory:
    def __init__(self, source):
        columns = {name: np.asarray(source[name]) for name in HISTORY_COLUMNS}
        order = np.lexsort((columns["txn_day"], columns["user_id"]))
        frame = {name: values[order] for name, values in columns.items()}
        del order, columns
        self.user_id = frame["user_id"]
        self.txn_day = frame["txn_day"]
        self.expire_day = frame["expire_day"]
        self.method = frame["payment_method_id"]
        self.plan_days = frame["payment_plan_days"]
        self.list_price = frame["plan_list_price"]
        self.paid = frame["actual_amount_paid"]
        self.auto_renew = frame["is_auto_renew"]
        self.is_cancel = frame["is_cancel"]
        self.key = self.user_id.astype(np.int64) * DAY_SCALE + self.txn_day
        self.p_paid = _prefix(self.paid, np.int64)
        self.p_plan_days = _prefix(self.plan_days, np.int64)
        self.p_auto_renew = _prefix(self.auto_renew, np.int32)
        self.p_cancel = _prefix(self.is_cancel, np.int32)
        self.p_discount = _prefix(self.paid < self.list_price, np.int32)


def _as_of(history, user_id, cutoff_day):
    keys = user_id.astype(np.int64) * DAY_SCALE + cutoff_day.astype(np.int64)
    upper = np.searchsorted(history.key, keys, side="left")
    lower = np.searchsorted(history.key, user_id.astype(np.int64) * DAY_SCALE, side="left")
    return lower, upper


def _clean_age(values, low, high):
    age = np.asarray(values, dtype=np.float32)
    bad = (age < low) | (age > high)
    age[bad] = np.nan
    return age


def _compute_block(config, history, block, members_lookup):
    user_id = block["user_id"].to_numpy().astype(np.int64)
    period_start = block["period_start_day"].to_numpy().astype(np.int64)
    lower, upper = _as_of(history, user_id, period_start)
    count = upper - lower
    has_history = count > 0
    last = np.where(has_history, upper - 1, 0)
    prev = np.where(count > 1, upper - 2, 0)
    has_prev = count > 1

    window_days = config.features.recent_window_periods * config.panel.period_days
    _, recent_lower = _as_of(history, user_id, period_start - window_days)
    recent_lower = np.maximum(recent_lower, lower)

    out = {}
    out["user_id"] = block["user_id"].to_numpy().astype(np.int32)
    out["period"] = block["period"].to_numpy().astype(np.int16)
    out["period_start_day"] = block["period_start_day"].to_numpy().astype(np.int32)
    out["event"] = block["event"].to_numpy().astype(np.int8)
    out["cause"] = block["cause"].to_numpy().astype(np.int8)

    origin = history.txn_day[lower]
    out["tenure_days"] = np.where(has_history, period_start - origin, np.nan)
    out["days_since_last_txn"] = np.where(has_history, period_start - history.txn_day[last], np.nan)
    expiry = history.expire_day[last]
    out["days_to_expiry"] = np.where(has_history, expiry - period_start, np.nan)
    out["membership_days_left"] = np.where(
        has_history, np.maximum(expiry - period_start, 0), np.nan
    )
    out["last_plan_days"] = np.where(has_history, history.plan_days[last], np.nan)
    list_price = history.list_price[last]
    paid = history.paid[last]
    out["last_list_price"] = np.where(has_history, list_price, np.nan)
    out["last_paid"] = np.where(has_history, paid, np.nan)
    out["last_discount_ratio"] = np.where(
        has_history, paid / np.maximum(list_price, 1), np.nan
    )
    delta = list_price - history.list_price[prev]
    out["price_delta"] = np.where(has_prev, delta, 0.0)

    n_txns = count.astype(np.float64)
    out["n_txns"] = n_txns
    cancels = history.p_cancel[upper] - history.p_cancel[lower]
    out["n_cancels"] = cancels
    auto = history.p_auto_renew[upper] - history.p_auto_renew[lower]
    out["auto_renew_share"] = np.where(has_history, auto / np.maximum(n_txns, 1), np.nan)
    discounted = history.p_discount[upper] - history.p_discount[lower]
    out["discount_share"] = np.where(has_history, discounted / np.maximum(n_txns, 1), np.nan)
    cum_paid = history.p_paid[upper] - history.p_paid[lower]
    out["cum_paid"] = cum_paid
    out["mean_paid"] = np.where(has_history, cum_paid / np.maximum(n_txns, 1), np.nan)
    out["cum_plan_days"] = history.p_plan_days[upper] - history.p_plan_days[lower]
    out["n_txns_recent"] = (upper - recent_lower).astype(np.float64)
    out["n_cancels_recent"] = history.p_cancel[upper] - history.p_cancel[recent_lower]
    out["paid_recent"] = history.p_paid[upper] - history.p_paid[recent_lower]

    out["last_is_auto_renew"] = np.where(has_history, history.auto_renew[last], 0)
    out["last_is_cancel"] = np.where(has_history, history.is_cancel[last], 0)
    out["price_increased"] = np.where(has_prev, delta > 0, False)
    out["method_changed"] = np.where(has_prev, history.method[last] != history.method[prev], False)
    out["plan_changed"] = np.where(
        has_prev, history.plan_days[last] != history.plan_days[prev], False
    )
    out["auto_renew_dropped"] = np.where(
        has_prev, (history.auto_renew[prev] == 1) & (history.auto_renew[last] == 0), False
    )
    out["last_payment_method_id"] = np.where(has_history, history.method[last], MISSING_CODE)

    member_rows = members_lookup.reindex(block["user_id"].to_numpy())
    has_member = member_rows["city"].notna().to_numpy()
    out["has_member_record"] = has_member
    for name in ("city", "registered_via", "gender_code"):
        values = member_rows[name].to_numpy(dtype=np.float64, na_value=MISSING_CODE)
        out[name] = np.where(has_member, values, MISSING_CODE)
    out["age"] = _clean_age(
        member_rows["bd"].to_numpy(dtype=np.float64, na_value=-999.0),
        config.features.age_min,
        config.features.age_max,
    )
    registration = member_rows["registration_day"].to_numpy(dtype=np.float64, na_value=-1.0)
    out["days_since_registration"] = np.where(
        registration > 0, period_start - registration, np.nan
    )

    stamps = period_start.astype("datetime64[D]")
    out["calendar_month"] = (stamps.astype("datetime64[M]").astype(np.int64) % 12 + 1)

    for name in NUMERIC_FEATURES:
        out[name] = np.asarray(out[name], dtype=np.float32)
    for name in BINARY_FEATURES:
        out[name] = np.asarray(out[name], dtype=np.int8)
    for name in CATEGORICAL_FEATURES:
        out[name] = np.asarray(out[name], dtype=np.int16)
    return pd.DataFrame({name: out[name] for name in KEY_COLUMNS + FEATURE_COLUMNS})


def _members_lookup(config):
    path = config.paths.interim_dir / "members.parquet"
    members = pd.read_parquet(path)
    return members.set_index("user_id")


def build(config, block_rows=1000000, verbose=True):
    history = load_history(config)
    panel = pd.read_parquet(config.paths.processed_dir / "panel.parquet")
    panel = panel.sort_values(["user_id", "period"], kind="stable").reset_index(drop=True)
    members_lookup = _members_lookup(config)

    target = config.paths.processed_dir / "features.parquet"
    writer = pq.ParquetWriter(target, _schema(), compression="zstd")
    boundaries = np.append(np.arange(0, len(panel), block_rows), len(panel))
    rows = 0
    try:
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop <= start:
                continue
            block = panel.iloc[start:stop]
            frame = _compute_block(config, history, block, members_lookup)
            writer.write_table(pa.Table.from_pandas(frame, schema=_schema(), preserve_index=False))
            rows += len(frame)
            del frame
            gc.collect()
            if verbose:
                print("features rows %d" % rows)
    finally:
        writer.close()
    return {"rows": rows}


def apply_rare_categories(frame, train_mask, min_count):
    if min_count <= 0:
        return frame, {}
    allowed = {}
    for name in CATEGORICAL_FEATURES:
        counts = frame.loc[train_mask, name].value_counts()
        keep = set(counts[counts >= min_count].index.tolist())
        allowed[name] = keep
        frame[name] = np.where(frame[name].isin(keep), frame[name], RARE_CODE).astype(np.int16)
    return frame, allowed


def run(config, verbose=True):
    return build(config, verbose=verbose)
