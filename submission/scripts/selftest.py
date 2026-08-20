#!/usr/bin/env python3
"""
Pre-flight check: dependencies, config, data resolution, the CadQuery worker,
tool registration, prompt assembly and credentials. Calls no API.

    python lw_solution_v2/scripts/selftest.py
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from harness import _bootstrap  # noqa: E402

DEFAULT_CONFIG = _bootstrap.solution_path("config", "lw_config.json")

FAILURES = []


def ok(label, detail=""):
    print(f"[  OK  ] {label}" + (f"  -  {detail}" if detail else ""))


def bad(label, detail=""):
    FAILURES.append(label)
    print(f"[ FAIL ] {label}" + (f"\n         {detail}" if detail else ""))


def warn(label, detail=""):
    print(f"[ WARN ] {label}" + (f"\n         {detail}" if detail else ""))


def check_eval_imports():
    """
    Can `src/scripts/run_all_benchmarks.py` be imported? That pulls in mongita,
    open3d, torch, transformers and probreg. Reported as a WARNING, not a
    failure: the harness runs and produces outputs without any of it, and you
    can score on another machine.
    """
    import importlib.util

    compat = _bootstrap.solution_path("compat")
    shimmed = [n for n in ("probreg",) if importlib.util.find_spec(n) is None]

    env = dict(os.environ)
    parts = [_bootstrap.REPO_ROOT] + ([compat] if shimmed else [])
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)

    probe = subprocess.run(
        [sys.executable, "-c",
         "import src.scripts.run_all_benchmarks, src.scripts.benchmark_evals.edit; "
         "print('eval imports ok')"],
        capture_output=True, text=True, cwd=_bootstrap.REPO_ROOT, env=env,
    )

    if probe.returncode == 0:
        detail = "via compat shim: " + ", ".join(shimmed) if shimmed else ""
        ok("evaluation pipeline imports", detail)
        return

    tail = (probe.stderr or "").strip().splitlines()
    missing = [ln for ln in tail if "ModuleNotFoundError" in ln]
    warn("evaluation pipeline cannot be imported yet",
         (missing[-1] if missing else tail[-1] if tail else "")
         + "\n         The harness still runs; you just cannot score locally yet."
           "\n         Install with: pip install -r lw_solution_v2/requirements.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    print("=" * 78)
    print("harness self-test")
    print("=" * 78)

    # --- 1. dependencies ---------------------------------------------------
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        ok("pandas + pyarrow importable")
    except Exception as exc:  # noqa: BLE001
        bad("pandas + pyarrow importable", f"{type(exc).__name__}: {exc}")

    worker_python = os.environ.get("LW_CQ_PYTHON") or sys.executable
    probe = subprocess.run(
        [worker_python, "-c", "import cadquery; print(cadquery.__version__)"],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        ok("cadquery importable in the worker interpreter",
           f"v{probe.stdout.strip()} ({osp.basename(worker_python)})")
    else:
        bad("cadquery importable in the worker interpreter",
            (probe.stderr or "").strip()[-400:]
            + "\n         Set LW_CQ_PYTHON to an interpreter that has cadquery.")

    # --- 2. config ---------------------------------------------------------
    from harness.runner import load_config, resolve_paths

    try:
        config = load_config(args.config)
        ok("config loads", args.config)
    except Exception as exc:  # noqa: BLE001
        bad("config loads", f"{type(exc).__name__}: {exc}")
        return report()

    block = resolve_paths(config)
    for key in ("parquet", "data_root"):
        path = block.get(key)
        exists = osp.exists(path) if path else False
        (ok if exists else bad)(f"{key} exists", path)

    user_id = block.get("user_id")
    model_config = (config.get("benchmark_models") or {}).get(user_id)
    if model_config:
        ok(f"benchmark_models[{user_id!r}]",
           f"{model_config.get('family')}/{model_config.get('model')}")
    else:
        bad(f"benchmark_models[{user_id!r}]",
            f"known ids: {', '.join(config.get('benchmark_models', {}))}")

    eval_users = (config.get("benchmark_eval_users") or {}).get("edit", [])
    if user_id in eval_users:
        ok(f"{user_id!r} is in benchmark_eval_users.edit")
    else:
        bad(f"{user_id!r} is in benchmark_eval_users.edit",
            "without this the eval pipeline will silently skip your results")

    # --- 3. tasks ----------------------------------------------------------
    from harness.io_adapter import load_tasks

    try:
        tasks = load_tasks(block["parquet"], block["data_root"],
                           image_dirs=block.get("image_dirs"),
                           difficulty=block.get("difficulty"))
    except Exception as exc:  # noqa: BLE001
        bad("parquet -> tasks", f"{type(exc).__name__}: {exc}")
        return report()

    if not tasks:
        bad("parquet -> tasks", "no usable rows")
        return report()

    with_images = sum(1 for t in tasks if t.input_images)
    ok("parquet -> tasks",
       f"{len(tasks)} tasks, {with_images} with pre-rendered views")

    missing = [t.short_id for t in tasks if not osp.exists(t.input_step)]
    if missing:
        bad("every input STEP resolves", f"missing: {missing[:5]}")
    else:
        ok("every input STEP resolves")

    if with_images < len(tasks):
        warn(f"{len(tasks) - with_images} task(s) have no pre-rendered views",
             "the agent still runs, but with less visual context")

    # --- 4/5. CadQuery worker ---------------------------------------------
    from harness.cq import client as cq_client

    sample = tasks[0]
    analysis = cq_client.analyze(sample.input_step)
    if analysis.get("ok"):
        rep = analysis["report"]
        ok("worker: analyze a real STEP",
           f"{osp.basename(sample.input_step)}  "
           f"bbox={(rep.get('bounding_box') or {}).get('size')}  "
           f"faces={(rep.get('counts') or {}).get('faces')}")
    else:
        bad("worker: analyze a real STEP", str(analysis.get("error"))[:400])

    scratch = osp.join(tempfile.gettempdir(), "lw_selftest")
    os.makedirs(scratch, exist_ok=True)
    script_path = osp.join(scratch, "candidate.py")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(
            "def my_cad_function(args):\n"
            "    import cadquery as cq\n"
            "    shape = cq.importers.importStep(args['input_file'])\n"
            "    print('loaded ok')\n"
            "    return shape\n"
        )

    execution = cq_client.execute(script_path, sample.input_step, scratch,
                                  views=["toprightiso"], image_size=400)
    if execution.get("ok"):
        images = execution.get("images") or []
        ok("worker: execute -> STEP", osp.basename(execution.get("step_path") or ""))
        if images:
            ok("worker: render PNG", f"{len(images)} image(s)")
        else:
            warn("worker: render PNG produced nothing",
                 "STEP/STL metrics still work, but the agent loses visual "
                 "self-inspection. On headless Linux run under xvfb-run.")
    else:
        bad("worker: execute a script", str(execution.get("error"))[:600])

    # --- 6. tools + prompt -------------------------------------------------
    from harness import prompts
    from harness.tools import tool_names

    names = tool_names()
    (ok if names else bad)("tools registered", ", ".join(names))
    ok("system prompt assembles", f"{len(prompts.build_system_prompt())} chars")
    ok("task instruction assembles",
       f"{len(prompts.build_task_instruction(sample.request_text))} chars")

    # --- 7. the evaluation pipeline can be imported ------------------------
    # Not needed to RUN the harness, only to SCORE it, so a failure here is a
    # warning: you can still produce outputs and score them later.
    check_eval_imports()

    # --- 8. credentials ----------------------------------------------------
    family = (model_config or {}).get("family")
    key_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(family)
    if key_env:
        if os.environ.get(key_env):
            ok(f"{key_env} is set")
        else:
            bad(f"{key_env} is not set", f"required for family={family}")

    return report()


def report():
    print("=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    print("Next:  python lw_solution_v2/scripts/run_harness.py --limit 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
