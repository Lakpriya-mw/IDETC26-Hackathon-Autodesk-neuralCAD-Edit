"""
Run the agent over a request parquet and write benchmark-shaped output.

Everything the loop needs comes from the config's `lw_harness` block.
"""

import json
import os
import os.path as osp
import time
import traceback
from typing import List, Optional

from harness import _bootstrap
from harness.agent import AgentSettings, EditAgent
from harness.cq import client as cq_client
from harness.io_adapter import (
    TaskSpec,
    estimate_cost,
    existing_request_ids,
    load_tasks,
    make_edit_dirs,
    run_output_root,
    write_result,
)
from harness.llm import build_client


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_paths(config: dict) -> dict:
    """Anchor relative paths in `lw_harness` to the repo root."""
    block = dict(config.get("lw_harness") or {})
    for key in ("data_root", "parquet", "output_dir", "work_dir"):
        if block.get(key):
            block[key] = _bootstrap.resolve(block[key])
    block["image_dirs"] = [
        _bootstrap.resolve(p) for p in (block.get("image_dirs") or []) if p
    ]
    if not block.get("data_root"):
        block["data_root"] = _bootstrap.resolve(
            config.get("storage_dir", {}).get("path", "data/edit_192_external")
        )
    return block


def run(config_path: str, user_id: Optional[str] = None, limit: Optional[int] = None,
        only: Optional[List[str]] = None, resume: bool = True,
        dry_run: bool = False, workers: Optional[int] = None) -> dict:
    config = load_config(config_path)
    block = resolve_paths(config)

    user_id = user_id or block.get("user_id")
    if not user_id:
        raise ValueError("No user_id: pass --userId or set lw_harness.user_id.")

    model_config = config["benchmark_models"].get(user_id)
    if model_config is None:
        raise ValueError(
            f"{user_id!r} is not in config['benchmark_models']. "
            f"Known: {', '.join(config.get('benchmark_models', {}))}"
        )

    parquet = block["parquet"]
    data_root = block["data_root"]
    output_dir = block["output_dir"]
    harness_name = block.get("harness_name", "lw_agentic")
    settings = AgentSettings.from_dict(block.get("agent"))
    workers = int(workers if workers is not None else block.get("workers", 1) or 1)

    print("=" * 78)
    print(f"harness   : {harness_name}")
    print(f"model     : {user_id}  ({model_config.get('family')}/{model_config.get('model')})")
    print(f"parquet   : {parquet}")
    print(f"data root : {data_root}")
    print(f"output    : {output_dir}")
    print(f"budget    : {settings.max_steps} steps / {settings.max_builds} builds per task")
    print(f"workers   : {workers}")
    print("=" * 78)

    tasks = load_tasks(parquet, data_root, view_names=settings.input_views,
                       image_dirs=block.get("image_dirs"),
                       difficulty=block.get("difficulty"))
    print(f"Loaded {len(tasks)} task(s) from the parquet.")

    if only:
        wanted = set(only)
        tasks = [t for t in tasks
                 if t.request_id in wanted or t.short_id in wanted]
        print(f"Filtered to {len(tasks)} task(s) by --only.")

    run_root = run_output_root(output_dir, harness_name, user_id, parquet)
    os.makedirs(run_root, exist_ok=True)

    done = existing_request_ids(run_root) if resume else set()
    if done:
        print(f"Resuming: {len(done)} request(s) already have output; skipping them.")

    # Outside models_dir, so the ingest crawler never sees intermediate builds.
    work_root = block.get("work_dir") or _bootstrap.solution_path(".work", harness_name)

    if dry_run:
        # Spell out what each task actually sends.
        payload = []
        if settings.bootstrap_brep_json:
            payload.append(f"brep-json digest (<={settings.brep_json_max_chars} chars)")
        if settings.bootstrap_views:
            payload.append(f"{len(settings.bootstrap_views)} generated view(s): "
                           f"{', '.join(settings.bootstrap_views)}")
        if settings.input_views:
            payload.append(f"{len(settings.input_views)} dataset view(s)")
        print("\nEach task's opening message carries: request text + "
              + (", ".join(payload) if payload else "no geometry context")
              + ".\nThe STEP file itself is never sent - only its path, which the "
                "generated script opens.\n")

        for task in tasks[: (limit or len(tasks))]:
            flag = "SKIP" if task.request_id in done else "RUN "
            print(f"  [{flag}] {task.short_id}  {task.request_text[:70]}")
        return {"dry_run": True, "n_tasks": len(tasks)}

    client = build_client(model_config)
    agent = EditAgent(client, model_config, settings)

    pending = [t for t in tasks if t.request_id not in done]
    skipped = len(tasks) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    summary = {"completed": 0, "fallback": 0, "failed": 0, "skipped": skipped,
               "cost": 0.0, "tasks": []}

    def run_one(position: int, task: TaskSpec) -> dict:
        """Run one task end to end and write its output folder."""
        tag = f"[{position}/{len(pending)}] {task.short_id}"
        print(f"\n{tag}\n  request: {task.request_text[:150]}\n"
              f"  input  : {osp.basename(task.input_step)} "
              f"({len(task.input_images)} pre-rendered views)")

        start = time.time()
        edit_id, brep_end_dir = make_edit_dirs(run_root, user_id, start)
        work_dir = osp.join(work_root, edit_id)

        try:
            result = agent.run(task, work_dir)
            error = result.error
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
            traceback.print_exc()
            result = None
            error = f"{type(exc).__name__}: {exc}"

        end = time.time()

        usage = dict(result.usage) if result else {}
        usage["cost_estimate"] = estimate_cost(usage, model_config)

        settings_dict = {
            "edit_request_id": task.request_id,
            "edit_id": edit_id,
            "start_time": start,
            "end_time": end,
            "isHuman": False,
            "userId": user_id,
            "token_counts": usage,
            # provenance; ignored by the ingest
            "lw_status": result.status if result else "crashed",
            "lw_used_fallback": bool(result.used_fallback) if result else False,
            "lw_builds": result.builds if result else 0,
            "lw_steps": result.steps if result else 0,
            "lw_harness": harness_name,
        }
        if error:
            settings_dict["lw_error"] = error

        written = write_result(
            brep_end_dir,
            settings_dict,
            result.step_path if result else None,
            image_size=block.get("output_image_size", 1024),
            image_ext=block.get("output_image_ext", "png"),
        )

        # Keep the winning script and the reasoning trace next to the output.
        if result:
            if result.script:
                with open(osp.join(brep_end_dir, "final_script.py"), "w",
                          encoding="utf-8") as fh:
                    fh.write(result.script)
            if block.get("save_transcripts", True):
                with open(osp.join(brep_end_dir, "transcript.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(result.transcript, fh, indent=2, default=str)

        status = written.get("lw_status", "crashed")
        print(f"{tag} -> {status}  builds={settings_dict['lw_builds']}  "
              f"{end - start:.0f}s  ${usage.get('cost_estimate', 0.0):.3f}")

        return {
            "request": task.request_id,
            "status": status,
            "failed_run": bool(written.get("failed_run")),
            "builds": settings_dict["lw_builds"],
            "seconds": round(end - start, 1),
            "cost": round(usage.get("cost_estimate", 0.0), 4),
        }

    def record(row: dict):
        if row["failed_run"]:
            summary["failed"] += 1
        elif row["status"] == "fallback":
            summary["fallback"] += 1
        else:
            summary["completed"] += 1
        summary["cost"] += row["cost"]
        summary["tasks"].append(row)

    if workers > 1 and len(pending) > 1:
        # Independent tasks, CAD in subprocesses: threads scale on API latency.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Running {len(pending)} task(s) across {workers} workers.")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, i, t): t
                for i, t in enumerate(pending, start=1)
            }
            for future in as_completed(futures):
                try:
                    record(future.result())
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                    summary["failed"] += 1
    else:
        for position, task in enumerate(pending, start=1):
            record(run_one(position, task))

    # Retire the per-thread CadQuery workers.
    cq_client.shutdown_all()

    print("\n" + "=" * 78)
    print(f"completed={summary['completed']}  fallback={summary['fallback']}  "
          f"failed={summary['failed']}  skipped={summary['skipped']}")
    print(f"estimated cost: ${summary['cost']:.2f}")
    print(f"output root   : {run_root}")
    print("=" * 78)

    with open(osp.join(run_root, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary
