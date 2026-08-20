"""
Input / output adapter - the contract with the organisers' pipeline.

Reads their request parquet, resolving `brep_start_path` against a configurable
data root, and writes exactly the folder shape their ingest crawls for:

    <output_dir>/<harness>/<userId>/<parquet_stem>/<edit_id>/brep_end/<ts>/
        settings.json  tmp.step  tmp.stl  tmp_<view>.png x7
"""

import json
import os
import os.path as osp
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

from harness import _bootstrap  # noqa: F401
from harness.cq import client as cq_client

# The 7 views the benchmark expects beside every output STEP.
REQUIRED_VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class TaskSpec:
    """One benchmark row, resolved to absolute paths."""

    request_id: str
    request_text: str
    input_step: str
    file_name: str = ""
    input_images: List[str] = field(default_factory=list)
    request_type: str = "edit"
    row_index: int = -1

    @property
    def short_id(self) -> str:
        return self.request_id.split("_")[0]


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------

def _flatten(value) -> List[str]:
    """`brep_start_path` arrives as a ragged array of arrays of strings."""
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        return [value]
    try:
        for item in value:
            out.extend(_flatten(item))
    except TypeError:
        out.append(str(value))
    return out


def find_input_images(step_path: str, views: Optional[List[str]] = None,
                      image_dirs: Optional[List[str]] = None) -> List[str]:
    """
    Pre-rendered views shipped beside a start BREP as `<stem>_<view>.jpg`.

    Searches beside the STEP first, then any `image_dirs`.
    """
    # None means "caller did not say" -> look for all of them.
    # [] means "caller explicitly wants none" -> do not go looking.
    if views is None:
        views = REQUIRED_VIEWS
    if not views:
        return []

    stem = osp.splitext(step_path)[0]
    base = osp.basename(stem)

    search_dirs = [osp.dirname(step_path)] + [d for d in (image_dirs or []) if d]

    found = []
    for view in views:
        hit = None
        for directory in search_dirs:
            for ext in IMAGE_EXTS:
                candidate = osp.join(directory, f"{base}_{view}{ext}")
                if osp.exists(candidate):
                    hit = candidate
                    break
            if hit:
                break
        if hit:
            found.append(hit)
    return found


def load_manifest(data_root: str) -> Optional[dict]:
    """
    request_id -> (difficulty, stem), for a difficulty-split data folder.

    None when absent, in which case `brep_start_path` is resolved literally.
    """
    import csv

    path = osp.join(data_root, "manifest.csv")
    if not osp.exists(path):
        return None

    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return None

    if not rows or "brep_filename_stem" not in rows[0]:
        print(f"  [warn] {path} has no 'brep_filename_stem' column - ignoring it")
        return None

    return {
        r["request_id"]: {
            "difficulty": r.get("difficulty", ""),
            "stem": r["brep_filename_stem"],
        }
        for r in rows if r.get("request_id") and r.get("brep_filename_stem")
    }


def load_tasks(parquet_path: str, data_root: str,
               view_names: Optional[List[str]] = None,
               image_dirs: Optional[List[str]] = None,
               difficulty: Optional[str] = None) -> List[TaskSpec]:
    """
    Parquet -> TaskSpecs, paths absolute.

    Resolves via `manifest.csv` when `data_root` has one, else by joining
    `brep_start_path`. `difficulty` needs a manifest - it is not a parquet column.
    """
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    manifest = load_manifest(data_root)
    tasks: List[TaskSpec] = []

    if manifest:
        print(f"  using manifest.csv in {data_root} ({len(manifest)} entries)")
    elif difficulty:
        print(f"  [warn] difficulty={difficulty!r} requested but no manifest.csv "
              f"in {data_root}; difficulty is not in the parquet, so the filter "
              f"cannot be applied")

    wanted = (difficulty or "").strip().lower() or None
    skipped_difficulty = 0

    for index, row in frame.iterrows():
        request_id = str(row.get("request"))
        entry = manifest.get(request_id) if manifest else None

        if wanted and entry and entry["difficulty"].lower() != wanted:
            skipped_difficulty += 1
            continue

        if entry:
            step_path = osp.abspath(
                osp.join(data_root, entry["difficulty"], f"{entry['stem']}.step")
            )
        else:
            paths = _flatten(row.get("brep_start_path"))
            step_paths = [p for p in paths if p.lower().endswith((".step", ".stp"))]
            if not step_paths:
                print(f"  [skip] {request_id}: no STEP in brep_start_path")
                continue
            step_path = step_paths[0]
            if not osp.isabs(step_path):
                step_path = osp.join(data_root, step_path)
            step_path = osp.abspath(step_path)

        if not osp.exists(step_path):
            print(f"  [skip] {request_id}: missing STEP {step_path}")
            continue

        text = row.get("request_text")
        if not isinstance(text, str) or not text.strip():
            print(f"  [skip] {row.get('request')}: no request_text")
            continue

        tasks.append(
            TaskSpec(
                request_id=str(row["request"]),
                request_text=text.strip(),
                input_step=step_path,
                file_name=str(row.get("file_name") or ""),
                input_images=find_input_images(step_path, view_names, image_dirs),
                request_type=str(row.get("request_type") or "edit"),
                row_index=int(index),
            )
        )

    if skipped_difficulty:
        print(f"  filtered out {skipped_difficulty} task(s) not of difficulty "
              f"{wanted!r}")

    return tasks


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------

def run_output_root(output_dir: str, harness_name: str, user_id: str,
                    parquet_path: str) -> str:
    """The `<output_dir>/<harness>/<userId>/<parquet_stem>` folder for a run."""
    stem = osp.splitext(osp.basename(parquet_path))[0]
    return osp.join(output_dir, harness_name, user_id, stem)


def existing_request_ids(run_root: str) -> set:
    """Request ids already written under `run_root`, for resume."""
    done = set()
    if not osp.isdir(run_root):
        return done
    for root, _dirs, files in os.walk(run_root):
        if "settings.json" not in files:
            continue
        try:
            with open(osp.join(root, "settings.json"), "r", encoding="utf-8") as fh:
                settings = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if settings.get("edit_request_id") and not settings.get("failed_run"):
            done.add(settings["edit_request_id"])
    return done


def make_edit_dirs(run_root: str, user_id: str, start_time: float):
    """Create `<edit_id>/brep_end/<start>/` and return (edit_id, that path)."""
    edit_id = f"{user_id}_{start_time}"
    brep_end_dir = osp.join(run_root, edit_id, "brep_end", str(start_time))
    os.makedirs(brep_end_dir, exist_ok=True)
    return edit_id, brep_end_dir


def write_result(brep_end_dir: str, settings: dict, step_source: Optional[str],
                 image_size: int = 1024, image_ext: str = "png") -> dict:
    """Place the STEP, derive STL + 7 views, write settings.json last."""
    settings = dict(settings)
    step_target = osp.join(brep_end_dir, "tmp.step")

    if step_source and osp.exists(step_source):
        if osp.abspath(step_source) != osp.abspath(step_target):
            shutil.copy(step_source, step_target)
    else:
        settings["failed_run"] = True

    if osp.exists(step_target):
        result = cq_client.convert(
            step_target, views=REQUIRED_VIEWS,
            image_ext=image_ext, image_size=image_size,
        )
        if not result.get("ok"):
            print(f"    [warn] STL/view export failed: {result.get('error')}")
            settings["failed_run"] = True
        settings["filename"] = step_target
    else:
        settings["filename"] = None
        settings["failed_run"] = True

    with open(osp.join(brep_end_dir, "settings.json"), "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=4, default=str)

    return settings


def estimate_cost(usage: dict, model_config: dict) -> float:
    """Cost estimate; counts cache reads at ~0.1x and writes at ~1.25x input."""
    def _f(key):
        return float(usage.get(key, 0) or 0)

    billed_input = (
        _f("input_tokens")
        + 0.10 * _f("cache_read_tokens")
        + 1.25 * _f("cache_write_tokens")
    )
    billed_output = _f("output_tokens") + _f("thinking_tokens")

    return (
        model_config.get("1m_token_cost_input", 0.0) * billed_input / 1e6
        + model_config.get("1m_token_cost_output", 0.0) * billed_output / 1e6
    )
