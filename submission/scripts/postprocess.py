#!/usr/bin/env python3
"""
Repair pass: ensure every output STEP has its .stl and 7 views.

`run_harness.py` already does this per task; run it if a render crashed or a
run was interrupted. Skips anything already complete.

    python lw_solution_v2/scripts/postprocess.py [--root DIR] [--force]
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from harness import _bootstrap  # noqa: E402
from harness.cq import client as cq_client  # noqa: E402
from harness.io_adapter import REQUIRED_VIEWS  # noqa: E402


def find_steps(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        if osp.basename(osp.dirname(dirpath)) != "brep_end":
            continue
        for name in filenames:
            if name.lower().endswith(".step"):
                yield osp.join(dirpath, name)


def is_complete(step_path: str, ext: str) -> bool:
    base = osp.splitext(step_path)[0]
    if not osp.exists(base + ".stl"):
        return False
    return all(osp.exists(f"{base}_{view}.{ext}") for view in REQUIRED_VIEWS)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="lw_solution_v2/outputs")
    parser.add_argument("--image-ext", default="png", choices=["png", "jpg"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true",
                        help="re-export even when outputs already exist")
    args = parser.parse_args()

    root = _bootstrap.resolve(args.root)
    if not osp.isdir(root):
        print(f"No such directory: {root}")
        return 1

    steps = sorted(find_steps(root))
    print(f"Found {len(steps)} output STEP file(s) under {root}")

    done = skipped = failed = 0
    for index, step_path in enumerate(steps, start=1):
        if not args.force and is_complete(step_path, args.image_ext):
            skipped += 1
            continue

        print(f"[{index}/{len(steps)}] {osp.relpath(step_path, root)}")
        result = cq_client.convert(
            step_path, views=REQUIRED_VIEWS,
            image_ext=args.image_ext, image_size=args.image_size,
        )
        if result.get("ok"):
            done += 1
            print(f"    stl + {len(result.get('images') or [])} views")
        else:
            failed += 1
            print(f"    FAILED: {result.get('error')}")

    print(f"\nexported={done}  already complete={skipped}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
