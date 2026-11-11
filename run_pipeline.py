"""Run the whole pipeline with one command: raw CSVs -> forecasts table.

Usage (from the project root, with the virtual environment active):
    python run_pipeline.py                 # all steps
    python run_pipeline.py --from dbt      # resume from a step (e.g. after a failure)
    python run_pipeline.py --only predict  # one step only

Steps, in order:
    ingest   CSVs in data/raw/ -> DuckDB raw schema   (ingestion/load_raw.py)
    dbt      staging, marts and features + all tests  (dbt build)
    predict  train the final model, write serving.forecasts (forecasting.predict)

Each step runs as a separate process; the pipeline stops at the first
failure, so later steps never run on half-built data. Model evaluation
(forecasting.train, forecasting.final_eval) is not part of it: it is a
one-off experiment, not the routine forecast refresh.

Close notebooks that have data/forecast.duckdb open first: DuckDB allows
only one writer at a time.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def dbt_executable() -> str:
    """The dbt installed next to this Python (the project's .venv), not any other copy."""
    found = shutil.which("dbt", path=str(Path(sys.executable).parent)) or shutil.which("dbt")
    if found is None:
        sys.exit("dbt not found: run `pip install -r requirements.txt` in the virtual environment")
    return found


def steps(config: dict) -> dict[str, tuple[list[str], Path]]:
    """Step name -> (command, folder to run it in)."""
    # config.yaml is the single source of truth: pass its scope to dbt
    dbt_vars = {**config["scope"], "horizon": config["forecast"]["horizon"]}
    return {
        "ingest": ([sys.executable, "ingestion/load_raw.py"], ROOT),
        "dbt": ([dbt_executable(), "build", "--profiles-dir", ".", "--vars", json.dumps(dbt_vars)],
                ROOT / "dbt_project"),
        "predict": ([sys.executable, "-m", "forecasting.predict"], ROOT),
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    all_steps = steps(config)
    names = list(all_steps)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", choices=names, help="start at this step")
    group.add_argument("--only", choices=names, help="run only this step")
    args = parser.parse_args()

    if args.only:
        selected = [args.only]
    else:
        selected = names[names.index(args.start):] if args.start else names

    timings = []
    for name in selected:
        command, cwd = all_steps[name]
        shown = [Path(command[0]).stem] + command[1:]          # "python ..." / "dbt ...", no full paths
        print(f"\n=== [{name}] {' '.join(shown)} ===", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=cwd)
        minutes = (time.perf_counter() - started) / 60
        if result.returncode != 0:
            sys.exit(f"\nStep '{name}' failed (exit code {result.returncode}) after {minutes:.1f} min. "
                     f"Fix it, then resume with: python run_pipeline.py --from {name}")
        timings.append((name, minutes))

    print("\n=== Pipeline finished ===")
    for name, minutes in timings:
        print(f"  {name:<8} {minutes:5.1f} min")


if __name__ == "__main__":
    main()
