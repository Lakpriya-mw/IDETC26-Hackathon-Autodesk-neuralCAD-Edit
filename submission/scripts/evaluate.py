"""
Ingest this harness's outputs and score them with the organisers' pipeline.

Nothing is reimplemented here; this calls their two scripts in order:

    src/scripts/build_instructions_db.py   ingest into the mongita DB
    src/scripts/run_all_benchmarks.py      chamfer / volume F1 / diff F1 + plots

Ingesting writes into the shared database, so a one-time pristine snapshot is
taken first and any run can be undone.

    python lw_solution_v2/scripts/evaluate.py [--ingest-only|--recompute|--no-backup]
"""

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from harness import _bootstrap  # noqa: E402

DEFAULT_CONFIG = _bootstrap.solution_path("config", "lw_config.json")


def backup_database(config_path: str) -> None:
    """One-time pristine snapshot, so a bad ingest is always undoable."""
    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    storage = _bootstrap.resolve(
        (config.get("storage_dir") or {}).get("path", "data/edit_192_external")
    )
    db_dir = osp.join(storage, "mongita_db")
    backup = db_dir + ".pristine_backup"

    if not osp.isdir(db_dir):
        return
    if osp.exists(backup):
        print(f"[db] pristine backup already exists: {backup}")
        return

    shutil.copytree(db_dir, backup)
    print(f"[db] pristine backup written to: {backup}")
    print(f"[db] to undo every ingest, delete '{db_dir}' and rename the backup back.")


def _check_optional_deps() -> None:
    """probreg is imported at module scope by the evaluation but never called."""
    import importlib.util

    if importlib.util.find_spec("probreg") is None:
        print("[warn] probreg is not installed. The evaluation imports it at "
              "module scope and will fail to start.\n"
              "       Install it with `pip install probreg` (needs a C++ "
              "toolchain on Windows).")


def call(script_rel: str, config_path: str) -> int:
    script = _bootstrap.repo_path(*script_rel.split("/"))
    cmd = [sys.executable, script, "--config", config_path]

    env = dict(os.environ)
    parts = [_bootstrap.REPO_ROOT]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)

    print(f"\n$ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=_bootstrap.REPO_ROOT, env=env)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--recompute", action="store_true",
                        help="set recompute_metrics=true for this run")
    parser.add_argument("--no-backup", action="store_true",
                        help="skip the one-time pristine snapshot of the database")
    args = parser.parse_args()

    config_path = osp.abspath(args.config)

    if not args.no_backup:
        backup_database(config_path)

    _check_optional_deps()

    if args.recompute:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        config["recompute_metrics"] = True
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=4)
        print("recompute_metrics set to true in the config.")

    if not args.eval_only:
        if call("src/scripts/build_instructions_db.py", config_path) != 0:
            print("Ingest failed - stopping.")
            return 1

    if args.ingest_only:
        return 0

    if call("src/scripts/run_all_benchmarks.py", config_path) != 0:
        print("Benchmark run failed.")
        return 1

    results = _bootstrap.repo_path("data", "edit_192_external", "results")
    print("\nResults written to:")
    for name in ("all_results.json", "metric_bar_facets.png", "cost_barplot.png"):
        print(f"  {osp.join(results, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
