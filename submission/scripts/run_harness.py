"""
Run the agentic harness over a request parquet.

    python lw_solution_v2/scripts/run_harness.py [--config PATH]

Flags: --limit N, --only <request_id>, --dry-run, --no-resume,
       --userId <id>, --workers N. Runs resume by default.
"""

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from harness.runner import run  # noqa: E402

DEFAULT_CONFIG = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "config", "lw_config.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--userId", default=None,
                        help="key in benchmark_models (default: lw_harness.user_id)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many newly-run tasks")
    parser.add_argument("--only", action="append", default=None,
                        help="run only this request id (repeatable)")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-run tasks that already have output")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the tasks and exit without calling any API")
    parser.add_argument("--workers", type=int, default=None,
                        help="run this many tasks concurrently "
                             "(default: lw_harness.workers, or 1)")
    args = parser.parse_args()

    run(
        config_path=args.config,
        user_id=args.userId,
        limit=args.limit,
        only=args.only,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
