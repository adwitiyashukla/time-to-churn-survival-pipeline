import dataclasses
import shutil
from pathlib import Path

import pytest

from churn_survival import features, ingest, panel, synth
from churn_survival.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def make_config(directory, **overrides):
    shutil.copy(ROOT / "config.toml", directory / "config.toml")
    config = load_config(directory / "config.toml")
    for section, values in overrides.items():
        config = dataclasses.replace(
            config, **{section: dataclasses.replace(getattr(config, section), **values)}
        )
    config.paths.ensure()
    return config


@pytest.fixture(scope="session")
def small_config(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("small")
    config = make_config(workspace, synthetic={"n_users": 4000})
    synth.generate(config, verbose=False)
    ingest.run(config, verbose=False)
    panel.run(config, verbose=False)
    features.run(config, verbose=False)
    return config


@pytest.fixture
def handmade_config(tmp_path):
    config = make_config(tmp_path)
    rows = [
        ("A", 41, 30, 149, 149, 1, 20160101, 20160131, 0),
        ("B", 41, 90, 417, 417, 1, 20160101, 20160401, 0),
        ("B", 41, 90, 417, 0, 1, 20160215, 20160215, 1),
        ("C", 41, 90, 417, 417, 1, 20170201, 20170502, 0),
        ("D", 41, 30, 149, 149, 0, 20160101, 20160131, 0),
        ("D", 41, 30, 149, 149, 0, 20160401, 20160501, 0),
    ]
    header = (
        "msno,payment_method_id,payment_plan_days,plan_list_price,actual_amount_paid,"
        "is_auto_renew,transaction_date,membership_expire_date,is_cancel"
    )
    lines = [header] + [",".join(str(value) for value in row) for row in rows]
    (config.paths.raw_dir / config.data.transactions_file).write_text("\n".join(lines) + "\n")
    members = ["msno,city,bd,gender,registered_via,registration_init_time"]
    for name in ("A", "B", "C", "D"):
        members.append("%s,1,25,male,7,20140101" % name)
    (config.paths.raw_dir / config.data.members_file).write_text("\n".join(members) + "\n")
    ingest.run(config, verbose=False)
    return config
