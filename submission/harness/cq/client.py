"""
Client side of the CadQuery worker.

The worker is persistent, one per thread: importing cadquery + matplotlib costs
~8s, several times the work in a typical job, so paying it per call meant a
14-step task spent ~110s starting interpreters. A worker that dies or hangs is
killed and replaced. Set LW_CQ_PERSISTENT=0 for one-shot mode.
"""

import atexit
import json
import os
import os.path as osp
import queue
import subprocess
import sys
import tempfile
import threading
import uuid

from harness import _bootstrap  # noqa: F401  (sys.path)

RUNNER = osp.join(osp.dirname(osp.abspath(__file__)), "cq_runner.py")

DEFAULT_TIMEOUT = 180
STARTUP_TIMEOUT = 120


def _python_executable() -> str:
    """Worker interpreter; override with LW_CQ_PYTHON."""
    return os.environ.get("LW_CQ_PYTHON") or sys.executable


def _worker_env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", _bootstrap.REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _persistent_enabled() -> bool:
    return os.environ.get("LW_CQ_PERSISTENT", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# persistent worker
# ---------------------------------------------------------------------------

# Tracked so none is orphaned at exit.
_ALL_WORKERS = set()
_WORKERS_LOCK = threading.Lock()


@atexit.register
def _kill_all_workers():
    """Kill, not negotiate - piping EXIT during finalization races teardown."""
    with _WORKERS_LOCK:
        workers = list(_ALL_WORKERS)
        _ALL_WORKERS.clear()
    for worker in workers:
        try:
            worker.proc.kill()
        except Exception:
            pass


def shutdown_all():
    """Stop every worker in every thread. Call at the end of a run."""
    with _WORKERS_LOCK:
        workers = list(_ALL_WORKERS)
    for worker in workers:
        worker.stop()


class _Worker:
    """One long-lived `cq_runner.py --serve` process, owned by one thread."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [_python_executable(), RUNNER, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=_worker_env(),
            cwd=_bootstrap.REPO_ROOT,
        )
        self._replies = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

        ready = self._await_reply(STARTUP_TIMEOUT)
        if ready != "READY":
            self.kill()
            raise RuntimeError(f"worker did not report READY (got {ready!r})")

        with _WORKERS_LOCK:
            _ALL_WORKERS.add(self)

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self._replies.put(line.strip())
        except Exception:
            pass
        finally:
            self._replies.put(None)  # stream closed => process gone

    def _await_reply(self, timeout):
        try:
            return self._replies.get(timeout=timeout)
        except queue.Empty:
            return "__TIMEOUT__"

    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(self, job_path: str, result_path: str, timeout: int) -> str:
        self.proc.stdin.write(f"{job_path}|{result_path}\n")
        self.proc.stdin.flush()
        return self._await_reply(timeout)

    def kill(self):
        with _WORKERS_LOCK:
            _ALL_WORKERS.discard(self)
        try:
            self.proc.kill()
        except Exception:
            pass

    def stop(self, grace: float = 3.0):
        """Ask the worker to exit; kill it if it will not."""
        with _WORKERS_LOCK:
            _ALL_WORKERS.discard(self)
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("EXIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=grace)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


_thread_local = threading.local()


def _get_worker():
    worker = getattr(_thread_local, "worker", None)
    if worker is not None and worker.alive():
        return worker
    try:
        worker = _Worker()
    except Exception as exc:  # noqa: BLE001 - fall back to one-shot
        print(f"    [cq] persistent worker unavailable ({exc}); using one-shot mode")
        _thread_local.worker = None
        return None
    _thread_local.worker = worker
    return worker


def _drop_worker():
    worker = getattr(_thread_local, "worker", None)
    if worker is not None:
        worker.kill()
    _thread_local.worker = None


def shutdown():
    """Stop this thread's worker, if any. Called when a task finishes."""
    worker = getattr(_thread_local, "worker", None)
    if worker is not None:
        worker.stop()
    _thread_local.worker = None


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit(job: dict, timeout: int = DEFAULT_TIMEOUT, scratch_dir: str = None) -> dict:
    """Run one worker job. Always returns a dict with an `ok` key."""
    scratch_dir = scratch_dir or tempfile.gettempdir()
    os.makedirs(scratch_dir, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    job_path = osp.join(scratch_dir, f"job_{token}.json")
    res_path = osp.join(scratch_dir, f"res_{token}.json")

    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump(job, fh, indent=2)

    if _persistent_enabled():
        result = _submit_persistent(job, job_path, res_path, timeout)
    else:
        result = _submit_oneshot(job_path, res_path, timeout)

    for path in (job_path, res_path):
        try:
            os.remove(path)
        except OSError:
            pass

    return result


def _read_result(res_path: str) -> dict:
    try:
        with open(res_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"unreadable worker result: {exc}"}


def _timeout_error(timeout: int) -> dict:
    return {
        "ok": False,
        "error": (
            f"CadQuery worker timed out after {timeout}s. The operation is too "
            f"expensive or looped - simplify it (fewer edges, coarser tolerance, "
            f"no full-part fillet in one call)."
        ),
        "timeout": True,
    }


def _submit_persistent(job, job_path, res_path, timeout):
    worker = _get_worker()
    if worker is None:
        return _submit_oneshot(job_path, res_path, timeout)

    try:
        reply = worker.run(job_path, res_path, timeout)
    except Exception as exc:  # noqa: BLE001 - broken pipe: worker died mid-write
        _drop_worker()
        return {"ok": False, "error": f"worker pipe failed: {exc}", "crashed": True}

    if reply == "__TIMEOUT__":
        # The worker is wedged on this job; it cannot be reused.
        _drop_worker()
        return _timeout_error(timeout)

    if reply is None:
        # stdout closed => a native OpenCascade crash took the process down.
        _drop_worker()
        if osp.exists(res_path):
            return _read_result(res_path)
        return {
            "ok": False,
            "error": ("CadQuery worker crashed without writing a result - most "
                      "likely a native OpenCascade fault on this geometry."),
            "crashed": True,
        }

    if reply.startswith("ERR"):
        return {"ok": False, "error": reply[4:].strip()}

    if not osp.exists(res_path):
        return {"ok": False, "error": "worker reported OK but wrote no result"}

    return _read_result(res_path)


def _submit_oneshot(job_path, res_path, timeout):
    cmd = [_python_executable(), RUNNER, "--job", job_path, "--result", res_path]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout, env=_worker_env(), cwd=_bootstrap.REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return _timeout_error(timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not start CadQuery worker: {exc}"}

    if not osp.exists(res_path):
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        return {
            "ok": False,
            "error": (f"CadQuery worker died (exit {proc.returncode}) without "
                      f"writing a result.\n{tail}"),
            "crashed": True,
        }
    return _read_result(res_path)


# ---------------------------------------------------------------------------
# convenience wrappers
# ---------------------------------------------------------------------------

def analyze(step_file: str, timeout: int = 120, scratch_dir: str = None) -> dict:
    return submit({"kind": "analyze", "step_file": step_file},
                  timeout=timeout, scratch_dir=scratch_dir)


def execute(script_file: str, input_file: str, output_dir: str,
            views=None, image_size: int = 800, timeout: int = DEFAULT_TIMEOUT) -> dict:
    return submit(
        {
            "kind": "execute",
            "script_file": script_file,
            "input_file": input_file,
            "output_dir": output_dir,
            "views": views or ["toprightiso"],
            "image_size": image_size,
        },
        timeout=timeout,
        scratch_dir=output_dir,
    )


def render(step_file: str, output_dir: str, views=None,
           prefix: str = "tmp", image_size: int = 800, timeout: int = 180) -> dict:
    return submit(
        {
            "kind": "render",
            "step_file": step_file,
            "output_dir": output_dir,
            "views": views or ["toprightiso"],
            "prefix": prefix,
            "image_size": image_size,
        },
        timeout=timeout,
        scratch_dir=output_dir,
    )


def step_to_views(step_file: str, output_dir: str, views=None, draw_bbox: bool = True,
                  label_faces: bool = True, draw_edges: bool = False,
                  resolution: int = 900, prefix: str = "view",
                  timeout: int = 300, label_ids=None) -> dict:
    """
    Rendered views with bounding-box overlay and JSON-matching face labels.

    `label_ids` tags exactly those faces instead of the bbox-touching envelope.
    """
    return submit(
        {
            "kind": "step_to_views",
            "step_file": step_file,
            "output_dir": output_dir,
            "views": views or ["iso_top_right"],
            "draw_bbox": draw_bbox,
            "label_faces": label_faces,
            "draw_edges": draw_edges,
            "resolution": resolution,
            "prefix": prefix,
            "label_ids": label_ids,
        },
        timeout=timeout,
        scratch_dir=output_dir,
    )


def step_to_json(step_file: str, json_path: str = None, include_edges: bool = True,
                 include_adjacency: bool = True, max_chars: int = 6000,
                 timeout: int = 300, **face_filters) -> dict:
    """
    Structured B-rep description: full JSON on disk, budgeted digest returned.

    `face_filters` narrows which faces the digest lists - face_type, min_area,
    max_area, min_radius, max_radius, ids, touching_only, offset.
    """
    job = {
        "kind": "step_to_json",
        "step_file": step_file,
        "json_path": json_path,
        "include_edges": include_edges,
        "include_adjacency": include_adjacency,
        "max_chars": max_chars,
    }
    job.update({k: v for k, v in face_filters.items() if v is not None})
    return submit(
        job,
        timeout=timeout,
        scratch_dir=osp.dirname(json_path) if json_path else None,
    )


def convert(step_file: str, views=None, image_ext: str = "png",
            image_size: int = 1024, timeout: int = 600) -> dict:
    return submit(
        {
            "kind": "convert",
            "step_file": step_file,
            "views": views,
            "image_ext": image_ext,
            "image_size": image_size,
        },
        timeout=timeout,
        scratch_dir=osp.dirname(step_file),
    )
