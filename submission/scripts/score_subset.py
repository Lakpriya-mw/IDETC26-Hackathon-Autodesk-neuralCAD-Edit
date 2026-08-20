#!/usr/bin/env python3
"""
Score a partial run and compare it against the baselines like-for-like.

`run_all_benchmarks.py` counts every unattempted request as 0.0, which is right
for the leaderboard but uninformative on a partial run. This reads the ratings
their pipeline wrote and reports the mean over the requests actually attempted,
the same baselines restricted to those requests, and the full-benchmark number.

Computes nothing itself. Run `evaluate.py` first, or pass --run-eval.

    python lw_solution_v2/scripts/score_subset.py [--run-eval] [--difficulty easy]
"""

import argparse
import json
import os
import os.path as osp
import subprocess
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from harness import _bootstrap  # noqa: E402

DEFAULT_CONFIG = _bootstrap.solution_path("config", "lw_config.json")

METRICS = ["chamfer_similarity_norm", "volume_f1", "diff_f1"]
METRIC_LABEL = {
    "chamfer_similarity_norm": "Chamfer sim",
    "volume_f1": "Volume F1",
    "diff_f1": "Diff F1",
}


def _mean(values):
    values = [v for v in values if v is not None and v == v]
    return sum(values) / len(values) if values else None


def _fmt(value):
    return f"{value:.3f}" if value is not None else "   -  "


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--userId", default=None,
                        help="which model to report (default: lw_harness.user_id)")
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard"],
                        help="restrict the report to one difficulty band")
    parser.add_argument("--run-eval", action="store_true",
                        help="run ingest + metrics first, instead of only reading")
    args = parser.parse_args()

    config_path = osp.abspath(args.config)

    if args.run_eval:
        script = _bootstrap.solution_path("scripts", "evaluate.py")
        env = dict(os.environ)
        env["PYTHONPATH"] = _bootstrap.REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        rc = subprocess.run([sys.executable, script, "--config", config_path],
                            cwd=_bootstrap.REPO_ROOT, env=env).returncode
        if rc != 0:
            print("evaluate.py failed - not reporting.")
            return 1

    # Imports deferred: they pull in the organisers' stack, which needs the
    # eval dependencies installed.
    from src.utils.db import DatabaseManager
    from src.utils.process_config import load_config
    from src.utils.visualise_results import parse_rating

    config = load_config(config_path)
    user_id = args.userId or (config.get("lw_harness") or {}).get("user_id")
    dbm = DatabaseManager(config)

    # ---- the request universe: the text-conditioned edits ------------------
    requests = {r["_id"]: r for r in dbm.requests.find({"request_type": "edit"})}
    if args.difficulty:
        requests = {k: v for k, v in requests.items()
                    if (v.get("difficulty") or "").lower() == args.difficulty}
    if not requests:
        print("No matching requests in the database.")
        return 1

    # ---- collect every scored edit ----------------------------------------
    # scores[user][request][metric] = value
    scores = {}
    for edit in dbm.edits.find({}):
        request_id = edit.get("request")
        if request_id not in requests:
            continue
        rating = dbm.ratings.find_one({"edit": edit["_id"], "user": "similarity_eval"})
        if not rating:
            continue
        metrics = parse_rating(rating)
        if not metrics:
            continue

        editor = edit.get("user")
        user_doc = dbm.users.find_one({"_id": editor}) or {}
        if editor == requests[request_id].get("user"):
            label = "gt human"
        elif user_doc.get("is_human", True) and editor != user_id:
            label = "other human"
        else:
            label = editor

        scores.setdefault(label, {}).setdefault(request_id, {}).update(metrics)

    mine = scores.get(user_id, {})
    attempted = sorted(mine)
    if not attempted:
        print(f"No scored edits found for {user_id!r}.")
        print("Run `python lw_solution_v2/scripts/evaluate.py` first, or pass --run-eval.")
        print(f"Users present in the database: {', '.join(sorted(scores))}")
        return 1

    band = f" [{args.difficulty}]" if args.difficulty else ""
    print("=" * 78)
    print(f"{user_id}{band}")
    print(f"attempted {len(attempted)} of {len(requests)} request(s)")
    print("=" * 78)

    # ---- per request -------------------------------------------------------
    print(f"\n{'request':<26} {'diff':<7} " +
          "  ".join(f"{METRIC_LABEL[m]:>12}" for m in METRICS))
    print("-" * 78)
    for request_id in attempted:
        row = mine[request_id]
        difficulty = (requests[request_id].get("difficulty") or "?")[:6]
        print(f"{request_id[:24]:<26} {difficulty:<7} " +
              "  ".join(f"{_fmt(row.get(m)):>12}" for m in METRICS))

    # ---- your two numbers --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'':<34}" + "  ".join(f"{METRIC_LABEL[m]:>12}" for m in METRICS))
    print("-" * 78)

    subset = {m: _mean([mine[r].get(m) for r in attempted]) for m in METRICS}
    print(f"{'YOUR MEAN over the ' + str(len(attempted)) + ' attempted':<34}" +
          "  ".join(f"{_fmt(subset[m]):>12}" for m in METRICS))

    full = {m: _mean([mine.get(r, {}).get(m) or 0.0 for r in requests]) for m in METRICS}
    print(f"{'full benchmark (missing = 0.0)':<34}" +
          "  ".join(f"{_fmt(full[m]):>12}" for m in METRICS))

    # ---- like-for-like comparison -----------------------------------------
    others = [u for u in scores if u != user_id]
    if others:
        print("\n" + "=" * 78)
        print(f"SAME {len(attempted)} REQUESTS, everyone else")
        print("-" * 78)
        print(f"{'':<34}" + "  ".join(f"{METRIC_LABEL[m]:>12}" for m in METRICS)
              + "   n")
        for other in sorted(others):
            rows = scores[other]
            covered = [r for r in attempted if r in rows]
            if not covered:
                continue
            means = {m: _mean([rows[r].get(m) for r in covered]) for m in METRICS}
            print(f"{other[:32]:<34}" +
                  "  ".join(f"{_fmt(means[m]):>12}" for m in METRICS)
                  + f"  {len(covered):>3}")

    print("\nAll three metrics are in [0, 1]; higher is better.")
    print("Compare yourself against the SAME-subset rows above, not against a "
          "48-task average.")

    # ---- persist, so the comparison survives the terminal ------------------
    report = {
        "user_id": user_id,
        "difficulty_filter": args.difficulty,
        "n_attempted": len(attempted),
        "n_requests_in_scope": len(requests),
        "metrics": METRICS,
        "subset_mean": subset,
        "full_benchmark_mean_missing_as_zero": full,
        "per_request": {
            r: {
                "difficulty": requests[r].get("difficulty"),
                "request_text": requests[r].get("text"),
                **{m: mine[r].get(m) for m in METRICS},
            }
            for r in attempted
        },
        "same_subset_comparison": {
            other: {
                "n": len([r for r in attempted if r in scores[other]]),
                **{m: _mean([scores[other][r].get(m)
                             for r in attempted if r in scores[other]])
                   for m in METRICS},
            }
            for other in sorted(others)
            if any(r in scores[other] for r in attempted)
        },
        "note": ("subset_mean covers only the requests actually attempted. "
                 "full_benchmark_mean_missing_as_zero is what "
                 "run_all_benchmarks.py reports and counts every unattempted "
                 "request as 0.0."),
    }

    out_dir = _bootstrap.solution_path("results")
    os.makedirs(out_dir, exist_ok=True)
    name = f"subset_scores{'_' + args.difficulty if args.difficulty else ''}.json"
    out_path = osp.join(out_dir, name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
