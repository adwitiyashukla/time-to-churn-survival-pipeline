from __future__ import annotations

import numpy as np
import pandas as pd

CAUSE_NONE = 0
CAUSE_VOLUNTARY = 1
CAUSE_PASSIVE = 2
SENTINEL = -1000000


def _load_for_spells(config):
    frame = pd.read_parquet(
        config.paths.interim_dir / "transactions.parquet",
        columns=["user_id", "txn_day", "expire_day", "is_cancel"],
    )
    user_id = frame["user_id"].to_numpy()
    txn_day = frame["txn_day"].to_numpy()
    expire_day = frame["expire_day"].to_numpy()
    is_cancel = frame["is_cancel"].to_numpy()
    del frame
    tie_break = is_cancel if config.panel.cancel_wins_same_day else -is_cancel.astype(np.int8)
    order = np.lexsort((tie_break, txn_day, user_id))
    return user_id[order], txn_day[order], expire_day[order], is_cancel[order]


def build_spells(config, verbose=True):
    user_id, txn_day, expire_day, is_cancel = _load_for_spells(config)
    row_count = len(user_id)
    if row_count == 0:
        raise ValueError("no transactions available to build spells from")

    grace = np.int64(config.panel.grace_days)
    observation_end = np.int64(
        np.datetime64(str(config.data.observation_end)).astype("datetime64[D]").astype(int)
    )

    boundary = np.empty(row_count, dtype=bool)
    boundary[0] = True
    np.not_equal(user_id[1:], user_id[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], row_count)
    del boundary

    has_next = np.zeros(row_count, dtype=bool)
    np.equal(user_id[:-1], user_id[1:], out=has_next[:-1])
    next_txn = np.zeros(row_count, dtype=np.int64)
    next_txn[:-1] = txn_day[1:]
    lapsed_here = has_next & (next_txn > expire_day + grace)
    del has_next, next_txn

    positions = np.arange(row_count, dtype=np.int64)
    first_lapse = np.minimum.reduceat(np.where(lapsed_here, positions, row_count), starts)
    del lapsed_here, positions

    saw_gap = first_lapse < row_count
    terminal = np.where(saw_gap, first_lapse, ends - 1)
    terminal_expire = expire_day[terminal].astype(np.int64)

    event = saw_gap | (terminal_expire + grace <= observation_end)
    voluntary = is_cancel[terminal] == 1
    event_day = np.where(voluntary, txn_day[terminal].astype(np.int64), terminal_expire)
    end_day = np.where(event, event_day, terminal_expire)
    origin = txn_day[starts].astype(np.int64) + 1

    cause = np.where(event, np.where(voluntary, CAUSE_VOLUNTARY, CAUSE_PASSIVE), CAUSE_NONE)
    spells = pd.DataFrame(
        {
            "user_id": user_id[starts].astype(np.int32),
            "origin_day": origin.astype(np.int32),
            "end_day": end_day.astype(np.int32),
            "duration_days": (end_day - origin).astype(np.int32),
            "event": event.astype(np.int8),
            "cause": cause.astype(np.int8),
        }
    )
    spells = spells.loc[spells["duration_days"] > 0].reset_index(drop=True)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    spells.to_parquet(config.paths.processed_dir / "spells.parquet", index=False, compression="zstd")
    if verbose:
        print(
            "spells %d events %d voluntary %d passive %d censored %d"
            % (
                len(spells),
                int(spells["event"].sum()),
                int((spells["cause"] == CAUSE_VOLUNTARY).sum()),
                int((spells["cause"] == CAUSE_PASSIVE).sum()),
                int((spells["event"] == 0).sum()),
            )
        )
    return spells


def build_panel(config, spells=None, verbose=True):
    if spells is None:
        spells = pd.read_parquet(config.paths.processed_dir / "spells.parquet")

    period_days = config.panel.period_days
    max_periods = config.panel.max_periods

    duration = spells["duration_days"].to_numpy().astype(np.int64)
    event = spells["event"].to_numpy().astype(np.int64)
    origin = spells["origin_day"].to_numpy().astype(np.int64)
    cause = spells["cause"].to_numpy().astype(np.int64)
    user_id = spells["user_id"].to_numpy()

    whole_periods = duration // period_days
    event_period = np.minimum(whole_periods, max_periods - 1)
    truncated = event & (whole_periods > max_periods - 1)
    counts = np.where(event == 1, event_period + 1, np.minimum(whole_periods, max_periods))
    counts = np.where(truncated == 1, max_periods, counts)
    counts = np.maximum(counts, 0)

    keep = counts >= config.panel.min_periods_observed
    counts = counts[keep]
    user_id = user_id[keep]
    origin = origin[keep]
    event = np.where(truncated[keep] == 1, 0, event[keep])
    cause = np.where(truncated[keep] == 1, CAUSE_NONE, cause[keep])
    event_period = event_period[keep]

    total = int(counts.sum())
    if total == 0:
        raise ValueError("panel is empty, check period_days and the observation window")

    starts = np.cumsum(counts) - counts
    row_user = np.repeat(np.arange(len(counts)), counts)
    period = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)

    panel = pd.DataFrame(
        {
            "user_id": user_id[row_user].astype(np.int32),
            "period": period.astype(np.int16),
            "period_start_day": (origin[row_user] + period * period_days).astype(np.int32),
            "event": ((event[row_user] == 1) & (period == event_period[row_user])).astype(np.int8),
        }
    )
    panel["cause"] = np.where(panel["event"].to_numpy() == 1, cause[row_user], CAUSE_NONE).astype(
        np.int8
    )
    panel.to_parquet(config.paths.processed_dir / "panel.parquet", index=False, compression="zstd")
    if verbose:
        print(
            "panel rows %d users %d events %d hazard %.4f"
            % (
                len(panel),
                len(counts),
                int(panel["event"].sum()),
                float(panel["event"].mean()),
            )
        )
    return panel


def run(config, verbose=True):
    spells = build_spells(config, verbose=verbose)
    panel = build_panel(config, spells=spells, verbose=verbose)
    return {"spells": len(spells), "panel_rows": len(panel), "events": int(panel["event"].sum())}
