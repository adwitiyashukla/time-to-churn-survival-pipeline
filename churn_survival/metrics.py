from __future__ import annotations

import numpy as np


def kaplan_meier(durations, events):
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)
    order = np.argsort(durations, kind="stable")
    durations = durations[order]
    events = events[order]
    times, starts = np.unique(durations, return_index=True)
    at_risk = len(durations) - starts
    failures = np.add.reduceat(events.astype(np.int64), starts)
    factor = np.where(at_risk > 0, 1.0 - failures / at_risk, 1.0)
    survival = np.cumprod(factor)
    return times, survival, at_risk, failures


def censoring_curve(durations, events):
    events = np.asarray(events).astype(bool)
    times, survival, _, _ = kaplan_meier(durations, ~events)
    return times, survival


class StepFunction:
    def __init__(self, times, values, left=1.0):
        self.times = np.asarray(times, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.left = float(left)

    def __call__(self, query):
        query = np.atleast_1d(np.asarray(query, dtype=np.float64))
        position = np.searchsorted(self.times, query, side="right") - 1
        out = np.where(position < 0, self.left, self.values[np.maximum(position, 0)])
        return out

    def before(self, query):
        query = np.atleast_1d(np.asarray(query, dtype=np.float64))
        position = np.searchsorted(self.times, query, side="left") - 1
        return np.where(position < 0, self.left, self.values[np.maximum(position, 0)])


def _guard(values, floor):
    return np.maximum(values, floor)


def brier_score(times, survival_matrix, durations, events, censor_fn, floor=1e-3):
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)
    scores = np.empty(len(times), dtype=np.float64)
    g_at_event = _guard(censor_fn.before(durations), floor)
    for index, time in enumerate(times):
        predicted = np.asarray(survival_matrix[:, index], dtype=np.float64)
        g_at_time = _guard(censor_fn(np.array([time]))[0], floor)
        early_event = (durations <= time) & events
        still_alive = durations > time
        contribution = np.zeros(len(durations), dtype=np.float64)
        contribution[early_event] = (predicted[early_event] ** 2) / g_at_event[early_event]
        contribution[still_alive] = ((1.0 - predicted[still_alive]) ** 2) / g_at_time
        scores[index] = contribution.mean()
    return scores


def integrated_brier_score(times, survival_matrix, durations, events, censor_fn, floor=1e-3):
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2:
        raise ValueError("integrated brier score needs at least two time points")
    scores = brier_score(times, survival_matrix, durations, events, censor_fn, floor=floor)
    return float(np.trapezoid(scores, times) / (times[-1] - times[0]))


class Fenwick:
    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, position, delta):
        position += 1
        while position <= self.size:
            self.tree[position] += delta
            position += position & (-position)

    def prefix(self, position):
        position += 1
        total = 0
        while position > 0:
            total += self.tree[position]
            position -= position & (-position)
        return int(total)


def uno_c_index(durations, events, risk, tau, censor_fn, floor=1e-3):
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)
    risk = np.asarray(risk, dtype=np.float64)

    order = np.argsort(durations, kind="stable")
    durations = durations[order]
    events = events[order]
    risk = risk[order]

    ranks = np.unique(risk)
    rank_of = np.searchsorted(ranks, risk)
    tree = Fenwick(len(ranks))
    for position in rank_of:
        tree.add(int(position), 1)
    remaining = len(durations)

    weights = 1.0 / _guard(censor_fn.before(durations), floor) ** 2
    numerator = 0.0
    denominator = 0.0

    index = 0
    total = len(durations)
    while index < total:
        stop = index
        while stop < total and durations[stop] == durations[index]:
            stop += 1
        for position in rank_of[index:stop]:
            tree.add(int(position), -1)
            remaining -= 1
        if durations[index] <= tau:
            for offset in range(index, stop):
                if not events[offset]:
                    continue
                rank = int(rank_of[offset])
                lower = tree.prefix(rank - 1) if rank > 0 else 0
                tied = tree.prefix(rank) - lower
                weight = weights[offset]
                numerator += weight * (lower + 0.5 * tied)
                denominator += weight * remaining
        index = stop
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def cumulative_dynamic_auc(time, risk, durations, events, censor_fn, floor=1e-3):
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)
    risk = np.asarray(risk, dtype=np.float64)

    cases = (durations <= time) & events
    controls = durations > time
    if cases.sum() == 0 or controls.sum() == 0:
        return float("nan")
    weights = 1.0 / _guard(censor_fn.before(durations[cases]), floor)
    control_risk = np.sort(risk[controls])
    case_risk = risk[cases]
    lower = np.searchsorted(control_risk, case_risk, side="left")
    upper = np.searchsorted(control_risk, case_risk, side="right")
    concordant = lower + 0.5 * (upper - lower)
    numerator = float(np.sum(weights * concordant))
    denominator = float(np.sum(weights) * len(control_risk))
    return numerator / denominator


def calibration_table(predicted_survival, durations, events, time, bins=10):
    predicted_survival = np.asarray(predicted_survival, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)
    quantiles = np.quantile(predicted_survival, np.linspace(0, 1, bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    edges = np.unique(quantiles)
    assignment = np.clip(np.searchsorted(edges, predicted_survival, side="left") - 1, 0, len(edges) - 2)
    rows = []
    for group in range(len(edges) - 1):
        mask = assignment == group
        if mask.sum() == 0:
            continue
        times, survival, _, _ = kaplan_meier(durations[mask], events[mask])
        curve = StepFunction(times, survival)
        rows.append(
            {
                "bin": group,
                "n": int(mask.sum()),
                "predicted_survival": float(predicted_survival[mask].mean()),
                "observed_survival": float(curve(np.array([time]))[0]),
            }
        )
    return rows


def calibration_error(rows):
    if not rows:
        return float("nan")
    total = sum(row["n"] for row in rows)
    return float(
        sum(
            row["n"] * abs(row["predicted_survival"] - row["observed_survival"])
            for row in rows
        )
        / total
    )
