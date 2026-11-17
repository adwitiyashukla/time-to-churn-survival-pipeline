from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


class ConfigError(ValueError):
    pass


def _table(raw, name):
    if name not in raw:
        raise ConfigError("config is missing section [%s]" % name)
    value = raw[name]
    if not isinstance(value, dict):
        raise ConfigError("section [%s] must be a table" % name)
    return value


def _get(table, key, kind, where):
    if key not in table:
        raise ConfigError("[%s] is missing key '%s'" % (where, key))
    value = table[key]
    if isinstance(value, bool) and kind is not bool:
        raise ConfigError("[%s].%s must be %s, got a boolean" % (where, key, kind.__name__))
    if kind is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, kind):
        raise ConfigError(
            "[%s].%s must be %s, got %s" % (where, key, kind.__name__, type(value).__name__)
        )
    return value


def _bounded(value, key, where, low=None, high=None):
    if low is not None and value < low:
        raise ConfigError("[%s].%s must be >= %s, got %s" % (where, key, low, value))
    if high is not None and value > high:
        raise ConfigError("[%s].%s must be <= %s, got %s" % (where, key, high, value))
    return value


def _num(table, key, kind, where, low=None, high=None):
    return _bounded(_get(table, key, kind, where), key, where, low, high)


def _choice(table, key, allowed, where):
    value = _get(table, key, str, where)
    if value not in allowed:
        raise ConfigError(
            "[%s].%s must be one of %s, got '%s'" % (where, key, ", ".join(allowed), value)
        )
    return value


def _day(table, key, where):
    text = _get(table, key, str, where)
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ConfigError(
            "[%s].%s must be a date as YYYY-MM-DD, got '%s'" % (where, key, text)
        ) from exc


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path

    def ensure(self):
        for target in (self.raw_dir, self.interim_dir, self.processed_dir, self.artifacts_dir):
            target.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    transactions_file: str
    members_file: str
    chunk_rows: int
    observation_start: date
    observation_end: date
    max_users: int


@dataclass(frozen=True)
class SyntheticConfig:
    enabled: bool
    n_users: int
    seed: int
    signup_start: date
    signup_end: date
    auto_renew_rate: float
    base_monthly_hazard: float
    tenure_hazard_decay: float
    auto_renew_hazard_multiplier: float
    price_increase_hazard_multiplier: float
    discount_hazard_multiplier: float
    voluntary_share_auto_renew: float
    voluntary_share_manual: float
    members_missing_rate: float


@dataclass(frozen=True)
class PanelConfig:
    period_days: int
    grace_days: int
    max_periods: int
    min_periods_observed: int
    cancel_wins_same_day: bool


@dataclass(frozen=True)
class FeatureConfig:
    recent_window_periods: int
    age_min: int
    age_max: int
    rare_category_min_count: int


@dataclass(frozen=True)
class SplitConfig:
    train_end: date
    valid_landmark: date
    test_landmark: date
    horizon_periods: int
    extended_horizon_periods: int
    min_window_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    fit_kaplan_meier: bool
    fit_cox: bool
    fit_cox_time_varying: bool
    fit_logistic_hazard: bool
    fit_boosted_hazard: bool
    fit_competing_risks: bool
    competing_risks_model: str
    train_max_rows: int
    cox_sample_users: int
    cox_penalizer: float
    logistic_c: float
    logistic_max_iter: int
    tenure_spline_knots: int
    boosted_max_iter: int
    boosted_learning_rate: float
    boosted_max_leaf_nodes: int
    boosted_min_samples_leaf: int
    boosted_l2: float
    boosted_early_stopping_rounds: int


@dataclass(frozen=True)
class EvaluateConfig:
    calibration_bins: int
    ipcw_clip_quantile: float
    bootstrap_rounds: int
    naive_baseline: bool


@dataclass(frozen=True)
class DecideConfig:
    contact_cost: float
    save_rate: float
    target_fraction: float
    period_margin_rate: float


@dataclass(frozen=True)
class RunConfig:
    seed: int
    n_jobs: int


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: DataConfig
    synthetic: SyntheticConfig
    panel: PanelConfig
    features: FeatureConfig
    split: SplitConfig
    models: ModelConfig
    evaluate: EvaluateConfig
    decide: DecideConfig
    run: RunConfig


def _paths(raw, root):
    table = _table(raw, "paths")
    return Paths(
        root=root,
        raw_dir=root / _get(table, "raw_dir", str, "paths"),
        interim_dir=root / _get(table, "interim_dir", str, "paths"),
        processed_dir=root / _get(table, "processed_dir", str, "paths"),
        artifacts_dir=root / _get(table, "artifacts_dir", str, "paths"),
    )


def _data(raw):
    table = _table(raw, "data")
    start = _day(table, "observation_start", "data")
    end = _day(table, "observation_end", "data")
    if start >= end:
        raise ConfigError("[data].observation_start must be earlier than observation_end")
    return DataConfig(
        transactions_file=_get(table, "transactions_file", str, "data"),
        members_file=_get(table, "members_file", str, "data"),
        chunk_rows=_num(table, "chunk_rows", int, "data", low=10000),
        observation_start=start,
        observation_end=end,
        max_users=_num(table, "max_users", int, "data", low=0),
    )


def _synthetic(raw, data):
    table = _table(raw, "synthetic")
    signup_start = _day(table, "signup_start", "synthetic")
    signup_end = _day(table, "signup_end", "synthetic")
    if signup_start >= signup_end:
        raise ConfigError("[synthetic].signup_start must be earlier than signup_end")
    if signup_start < data.observation_start or signup_end > data.observation_end:
        raise ConfigError("[synthetic] signup window must sit inside the observation window")
    return SyntheticConfig(
        enabled=_get(table, "enabled", bool, "synthetic"),
        n_users=_num(table, "n_users", int, "synthetic", low=1000),
        seed=_num(table, "seed", int, "synthetic", low=0),
        signup_start=signup_start,
        signup_end=signup_end,
        auto_renew_rate=_num(table, "auto_renew_rate", float, "synthetic", low=0.0, high=1.0),
        base_monthly_hazard=_num(
            table, "base_monthly_hazard", float, "synthetic", low=0.001, high=0.5
        ),
        tenure_hazard_decay=_num(
            table, "tenure_hazard_decay", float, "synthetic", low=0.0, high=1.0
        ),
        auto_renew_hazard_multiplier=_num(
            table, "auto_renew_hazard_multiplier", float, "synthetic", low=0.01, high=10.0
        ),
        price_increase_hazard_multiplier=_num(
            table, "price_increase_hazard_multiplier", float, "synthetic", low=0.01, high=10.0
        ),
        discount_hazard_multiplier=_num(
            table, "discount_hazard_multiplier", float, "synthetic", low=0.01, high=10.0
        ),
        voluntary_share_auto_renew=_num(
            table, "voluntary_share_auto_renew", float, "synthetic", low=0.0, high=1.0
        ),
        voluntary_share_manual=_num(
            table, "voluntary_share_manual", float, "synthetic", low=0.0, high=1.0
        ),
        members_missing_rate=_num(
            table, "members_missing_rate", float, "synthetic", low=0.0, high=0.5
        ),
    )


def _panel(raw):
    table = _table(raw, "panel")
    return PanelConfig(
        period_days=_num(table, "period_days", int, "panel", low=1, high=365),
        grace_days=_num(table, "grace_days", int, "panel", low=0, high=365),
        max_periods=_num(table, "max_periods", int, "panel", low=2, high=200),
        min_periods_observed=_num(table, "min_periods_observed", int, "panel", low=1),
        cancel_wins_same_day=_get(table, "cancel_wins_same_day", bool, "panel"),
    )


def _features(raw):
    table = _table(raw, "features")
    age_min = _num(table, "age_min", int, "features", low=0, high=120)
    age_max = _num(table, "age_max", int, "features", low=0, high=120)
    if age_min >= age_max:
        raise ConfigError("[features].age_min must be smaller than age_max")
    return FeatureConfig(
        recent_window_periods=_num(table, "recent_window_periods", int, "features", low=1),
        age_min=age_min,
        age_max=age_max,
        rare_category_min_count=_num(table, "rare_category_min_count", int, "features", low=0),
    )


def _split(raw, data, panel):
    table = _table(raw, "split")
    train_end = _day(table, "train_end", "split")
    valid_landmark = _day(table, "valid_landmark", "split")
    test_landmark = _day(table, "test_landmark", "split")
    horizon = _num(table, "horizon_periods", int, "split", low=1, high=panel.max_periods)
    if not train_end < valid_landmark < test_landmark:
        raise ConfigError(
            "[split] requires train_end < valid_landmark < test_landmark"
        )
    if train_end <= data.observation_start:
        raise ConfigError("[split].train_end must fall after [data].observation_start")
    horizon_days = horizon * panel.period_days
    extended = _num(
        table, "extended_horizon_periods", int, "split", low=horizon, high=panel.max_periods
    )
    fraction = _num(table, "min_window_fraction", float, "split", low=0.1, high=1.0)
    window = (data.observation_end - test_landmark).days
    if window < max(int(horizon_days * fraction), panel.period_days):
        raise ConfigError(
            "[split].test_landmark leaves only %d days of follow up, which is below "
            "min_window_fraction of the %d day horizon" % (window, horizon_days)
        )
    if (test_landmark - valid_landmark).days < horizon_days:
        raise ConfigError(
            "[split].valid_landmark follow up would overlap test_landmark; leave at least %d days"
            % horizon_days
        )
    return SplitConfig(
        train_end=train_end,
        valid_landmark=valid_landmark,
        test_landmark=test_landmark,
        horizon_periods=horizon,
        extended_horizon_periods=extended,
        min_window_fraction=fraction,
    )


def _models(raw):
    table = _table(raw, "models")
    return ModelConfig(
        fit_kaplan_meier=_get(table, "fit_kaplan_meier", bool, "models"),
        fit_cox=_get(table, "fit_cox", bool, "models"),
        fit_cox_time_varying=_get(table, "fit_cox_time_varying", bool, "models"),
        fit_logistic_hazard=_get(table, "fit_logistic_hazard", bool, "models"),
        fit_boosted_hazard=_get(table, "fit_boosted_hazard", bool, "models"),
        fit_competing_risks=_get(table, "fit_competing_risks", bool, "models"),
        competing_risks_model=_choice(
            table, "competing_risks_model", ("logistic", "boosted"), "models"
        ),
        train_max_rows=_num(table, "train_max_rows", int, "models", low=0),
        cox_sample_users=_num(table, "cox_sample_users", int, "models", low=500),
        cox_penalizer=_num(table, "cox_penalizer", float, "models", low=0.0, high=100.0),
        logistic_c=_num(table, "logistic_c", float, "models", low=1e-6, high=1e6),
        logistic_max_iter=_num(table, "logistic_max_iter", int, "models", low=10),
        tenure_spline_knots=_num(table, "tenure_spline_knots", int, "models", low=3, high=40),
        boosted_max_iter=_num(table, "boosted_max_iter", int, "models", low=10),
        boosted_learning_rate=_num(
            table, "boosted_learning_rate", float, "models", low=1e-4, high=1.0
        ),
        boosted_max_leaf_nodes=_num(table, "boosted_max_leaf_nodes", int, "models", low=2),
        boosted_min_samples_leaf=_num(table, "boosted_min_samples_leaf", int, "models", low=1),
        boosted_l2=_num(table, "boosted_l2", float, "models", low=0.0),
        boosted_early_stopping_rounds=_num(
            table, "boosted_early_stopping_rounds", int, "models", low=1
        ),
    )


def _evaluate(raw):
    table = _table(raw, "evaluate")
    return EvaluateConfig(
        calibration_bins=_num(table, "calibration_bins", int, "evaluate", low=2, high=100),
        ipcw_clip_quantile=_num(
            table, "ipcw_clip_quantile", float, "evaluate", low=0.5, high=1.0
        ),
        bootstrap_rounds=_num(table, "bootstrap_rounds", int, "evaluate", low=0, high=10000),
        naive_baseline=_get(table, "naive_baseline", bool, "evaluate"),
    )


def _decide(raw):
    table = _table(raw, "decide")
    return DecideConfig(
        contact_cost=_num(table, "contact_cost", float, "decide", low=0.0),
        save_rate=_num(table, "save_rate", float, "decide", low=0.0, high=1.0),
        target_fraction=_num(table, "target_fraction", float, "decide", low=0.001, high=1.0),
        period_margin_rate=_num(table, "period_margin_rate", float, "decide", low=0.0, high=1.0),
    )


def _run(raw):
    table = _table(raw, "run")
    return RunConfig(
        seed=_num(table, "seed", int, "run", low=0),
        n_jobs=_num(table, "n_jobs", int, "run", low=1, high=256),
    )


def load_config(path="config.toml"):
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError("no config file at %s" % config_path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    data = _data(raw)
    panel = _panel(raw)
    return Config(
        paths=_paths(raw, root),
        data=data,
        synthetic=_synthetic(raw, data),
        panel=panel,
        features=_features(raw),
        split=_split(raw, data, panel),
        models=_models(raw),
        evaluate=_evaluate(raw),
        decide=_decide(raw),
        run=_run(raw),
    )
