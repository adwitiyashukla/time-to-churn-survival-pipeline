from __future__ import annotations

import argparse

from . import decide, evaluate, features, ingest, panel, synth
from .config import load_config

STAGES = ["synth", "ingest", "panel", "features", "evaluate", "decide"]


def _run_stage(name, config, verbose):
    if name == "synth":
        return synth.generate(config, verbose=verbose)
    if name == "ingest":
        return ingest.run(config, verbose=verbose)
    if name == "panel":
        return panel.run(config, verbose=verbose)
    if name == "features":
        return features.run(config, verbose=verbose)
    if name == "evaluate":
        return evaluate.run(config, verbose=verbose)
    if name == "decide":
        return decide.run(config, verbose=verbose)
    raise ValueError("unknown stage %s" % name)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="churn_survival")
    parser.add_argument("stage", choices=STAGES + ["all"])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.paths.ensure()
    verbose = not args.quiet

    stages = STAGES[1:] if args.stage == "all" else [args.stage]
    for name in stages:
        if verbose:
            print("== %s ==" % name)
        _run_stage(name, config, verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
