from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, RARE_CODE


def to_day(value):
    return int(np.datetime64(str(value)).astype("datetime64[D]").astype(int))


def _read(config, filters):
    path = config.paths.processed_dir / "features.parquet"
    return pq.read_table(path, filters=filters).to_pandas()


def load_training(config, seed=None, batch_rows=500000):
    cutoff = to_day(config.split.train_end)
    path = config.paths.processed_dir / "features.parquet"
    source = pq.ParquetFile(path)
    starts = pq.read_table(path, columns=["period_start_day"]).column(0).to_numpy()
    eligible = int((starts <= cutoff).sum())
    del starts
    if eligible == 0:
        raise ValueError("no panel rows fall on or before [split].train_end")
    cap = config.models.train_max_rows
    keep_rate = 1.0 if not cap or eligible <= cap else cap / eligible
    rng = np.random.default_rng(config.run.seed if seed is None else seed)
    blocks = []
    for batch in source.iter_batches(batch_size=batch_rows):
        block = batch.to_pandas()
        block = block.loc[block["period_start_day"].to_numpy() <= cutoff]
        if block.empty:
            continue
        if keep_rate < 1.0:
            block = block.loc[rng.random(len(block)) < keep_rate]
            if block.empty:
                continue
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def load_landmark_window(config, landmark):
    day = to_day(landmark)
    lower = day - config.panel.period_days
    return _read(
        config,
        [("period_start_day", ">", lower), ("period_start_day", "<=", day)],
    )


def load_features(config):
    return _read(config, None)


def training_frame(config, frame):
    cutoff = to_day(config.split.train_end)
    return frame.loc[frame["period_start_day"].to_numpy() <= cutoff].reset_index(drop=True)


def fit_rare(config, frame):
    minimum = config.features.rare_category_min_count
    if minimum <= 0:
        return {}
    allowed = {}
    for name in CATEGORICAL_FEATURES:
        counts = frame[name].value_counts()
        allowed[name] = counts[counts >= minimum].index.to_numpy()
    return allowed


def apply_rare(frame, allowed):
    for name, keep in allowed.items():
        values = frame[name].to_numpy()
        frame[name] = np.where(np.isin(values, keep), values, RARE_CODE).astype(np.int16)
    return frame


def collapse_rare(config, frame, train_mask):
    allowed = fit_rare(config, frame.loc[train_mask])
    return apply_rare(frame, allowed), allowed


def landmark_cohort(config, window, spells, landmark, window_days=None):
    landmark_day = to_day(landmark)
    horizon_days = config.split.horizon_periods * config.panel.period_days
    if window_days is None:
        window_days = int(np.iinfo(np.int32).max)

    rows = window.loc[window["period_start_day"].to_numpy() <= landmark_day]
    rows = rows.sort_values(["user_id", "period_start_day"], kind="stable")
    latest = rows.groupby("user_id", sort=True).tail(1)

    spell_lookup = spells.set_index("user_id")[["origin_day", "end_day", "event", "cause"]]
    spell_lookup = spell_lookup.rename(
        columns={"event": "spell_event", "cause": "spell_cause"}
    )
    joined = latest.drop(columns=["event", "cause"]).join(spell_lookup, on="user_id")

    at_risk = joined["end_day"].to_numpy() > landmark_day
    joined = joined.loc[at_risk].reset_index(drop=True)

    raw_duration = joined["end_day"].to_numpy() - landmark_day
    observed = joined["spell_event"].to_numpy().astype(bool)
    beyond_window = raw_duration > window_days
    duration = np.minimum(raw_duration, window_days)
    event = observed & ~beyond_window
    cause = np.where(event, joined["spell_cause"].to_numpy(), 0)

    joined["duration"] = duration.astype(np.int32)
    joined["event"] = event.astype(np.int8)
    joined["cause"] = cause.astype(np.int8)
    joined["landmark_day"] = landmark_day
    joined["horizon_days"] = horizon_days
    joined["window_days"] = window_days
    return joined.reset_index(drop=True)


def design_matrix(frame, columns=None):
    columns = columns or FEATURE_COLUMNS
    return frame[columns].to_numpy(dtype=np.float32, na_value=np.nan)
