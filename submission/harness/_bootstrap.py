"""
Path bootstrap: makes the repo root and this folder importable regardless of
the working directory. Import first in every entry point.
"""

import os.path as osp
import sys

# .../lw_solution_v2/harness/_bootstrap.py -> .../lw_solution_v2 -> repo root
HARNESS_DIR = osp.dirname(osp.abspath(__file__))
SOLUTION_ROOT = osp.dirname(HARNESS_DIR)
REPO_ROOT = osp.dirname(SOLUTION_ROOT)

for _p in (SOLUTION_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def repo_path(*parts: str) -> str:
    """Absolute path relative to the original repo root."""
    return osp.join(REPO_ROOT, *parts)


def solution_path(*parts: str) -> str:
    """Absolute path relative to lw_solution_v2/."""
    return osp.join(SOLUTION_ROOT, *parts)


def resolve(path: str) -> str:
    """Absolute paths pass through; relative ones anchor to the repo root,
    matching `src/utils/process_config.py`."""
    if not path:
        return path
    expanded = osp.expanduser(path)
    if osp.isabs(expanded):
        return expanded
    return osp.abspath(osp.join(REPO_ROOT, expanded))
