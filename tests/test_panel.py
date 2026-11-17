import pandas as pd

from churn_survival import panel
from churn_survival.ingest import yyyymmdd_to_day


def _spells(config):
    frame = panel.build_spells(config, verbose=False)
    users = pd.read_parquet(config.paths.interim_dir / "users.parquet")
    return frame.merge(users, on="user_id").set_index("msno")


def _day(value):
    return int(yyyymmdd_to_day([value])[0][0])


def test_passive_lapse_is_an_event_at_expiry(handmade_config):
    spells = _spells(handmade_config)
    row = spells.loc["A"]
    assert row["event"] == 1
    assert row["cause"] == panel.CAUSE_PASSIVE
    assert row["end_day"] == _day(20160131)


def test_cancellation_stops_the_clock_at_the_cancel_date(handmade_config):
    spells = _spells(handmade_config)
    row = spells.loc["B"]
    assert row["event"] == 1
    assert row["cause"] == panel.CAUSE_VOLUNTARY
    assert row["end_day"] == _day(20160215)


def test_membership_running_past_the_window_is_censored(handmade_config):
    spells = _spells(handmade_config)
    row = spells.loc["C"]
    assert row["event"] == 0
    assert row["cause"] == panel.CAUSE_NONE
    assert row["end_day"] == _day(20170502)


def test_only_the_first_lapse_counts(handmade_config):
    spells = _spells(handmade_config)
    row = spells.loc["D"]
    assert row["event"] == 1
    assert row["end_day"] == _day(20160131)


def test_panel_rows_match_observed_periods(handmade_config):
    spells = panel.build_spells(handmade_config, verbose=False)
    rows = panel.build_panel(handmade_config, spells=spells, verbose=False)
    period_days = handmade_config.panel.period_days
    counts = rows.groupby("user_id").size()
    for _, spell in spells.iterrows():
        whole = spell["duration_days"] // period_days
        expected = whole + 1 if spell["event"] == 1 else whole
        assert counts.get(spell["user_id"], 0) == expected


def test_every_event_lands_on_the_last_row_of_its_spell(small_config):
    rows = pd.read_parquet(small_config.paths.processed_dir / "panel.parquet")
    last = rows.groupby("user_id")["period"].transform("max")
    assert ((rows["event"] == 0) | (rows["period"] == last)).all()
    assert rows.groupby("user_id")["event"].sum().max() <= 1
