"""
CadQuery worker process, driven by a JSON job file.

Every CadQuery/OpenCascade call runs here so a native fault or an infinite
boolean loop costs one job rather than the run, and the agent loop carries no
CAD imports.

    python cq_runner.py --job job.json --result result.json
    python cq_runner.py --serve            # persistent mode, jobs on stdin

Jobs: analyze | execute | render | convert | step_to_views | step_to_json.
Every result is {"ok": bool, "error": str|None, ...payload}.
"""

import argparse
import io
import json
import os
import os.path as osp
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout  # noqa: F401

# Make lw_solution_v2/ and the repo root importable regardless of how we are launched.
_HERE = osp.dirname(osp.abspath(__file__))
_SOLUTION_ROOT = osp.dirname(osp.dirname(_HERE))
_REPO_ROOT = osp.dirname(_SOLUTION_ROOT)
for _p in (_SOLUTION_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cadquery as cq  # noqa: E402
from cadquery import exporters  # noqa: E402

from src.utils.cadquery_rendering import VIEW_PROJECTIONS, render_to_png  # noqa: E402

DEFAULT_VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]


# ----------------------------------------------------------------------------
# shape helpers
# ----------------------------------------------------------------------------

def to_shape(result):
    """Normalise a Workplane / Assembly / Shape into a single cq Shape."""
    if result is None:
        return None
    if isinstance(result, cq.Assembly):
        return result.toCompound()
    if hasattr(result, "val"):
        val = result.val()
        # A Workplane holding several objects: fuse them so metrics are global.
        objs = [o for o in getattr(result, "objects", []) if hasattr(o, "wrapped")]
        if len(objs) > 1:
            try:
                return cq.Compound.makeCompound(objs)
            except Exception:
                return val
        return val
    return result


def _round(value, nd=4):
    try:
        return round(float(value), nd)
    except Exception:
        return None


def _cylinder_of(face):
    """(radius, axis_dir, axis_location) for a cylindrical face, else None."""
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        adaptor = BRepAdaptor_Surface(face.wrapped)
        cyl = adaptor.Cylinder()
        axis = cyl.Axis()
        d = axis.Direction()
        p = axis.Location()
        return (
            _round(cyl.Radius(), 4),
            [_round(d.X(), 3), _round(d.Y(), 3), _round(d.Z(), 3)],
            [_round(p.X(), 3), _round(p.Y(), 3), _round(p.Z(), 3)],
        )
    except Exception:
        return None


def analyze_shape(shape, max_items=40):
    """Compact description: bbox, volume, counts, hole radii and centres."""
    if shape is None:
        return {"error": "shape is None"}

    report = {}

    try:
        bb = shape.BoundingBox()
        report["bounding_box"] = {
            "xmin": _round(bb.xmin), "xmax": _round(bb.xmax),
            "ymin": _round(bb.ymin), "ymax": _round(bb.ymax),
            "zmin": _round(bb.zmin), "zmax": _round(bb.zmax),
            "size": [_round(bb.xlen), _round(bb.ylen), _round(bb.zlen)],
            "center": [_round(bb.center.x), _round(bb.center.y), _round(bb.center.z)],
            "diagonal": _round(bb.DiagonalLength),
        }
    except Exception as exc:
        report["bounding_box"] = {"error": str(exc)}

    for key, fn in (("volume", "Volume"), ("surface_area", "Area")):
        try:
            report[key] = _round(getattr(shape, fn)(), 4)
        except Exception:
            report[key] = None

    try:
        report["is_valid"] = bool(shape.isValid())
    except Exception:
        report["is_valid"] = None

    try:
        faces = shape.Faces()
        edges = shape.Edges()
        report["counts"] = {
            "solids": len(shape.Solids()),
            "shells": len(shape.Shells()),
            "faces": len(faces),
            "edges": len(edges),
            "vertices": len(shape.Vertices()),
        }
    except Exception as exc:
        report["counts"] = {"error": str(exc)}
        return report

    # --- face type histogram -------------------------------------------------
    face_types = {}
    for f in faces:
        try:
            face_types[f.geomType()] = face_types.get(f.geomType(), 0) + 1
        except Exception:
            face_types["UNKNOWN"] = face_types.get("UNKNOWN", 0) + 1
    report["face_types"] = face_types

    # --- cylindrical faces grouped by radius (holes / bosses / fillets) ------
    cyl_groups = {}
    for f in faces:
        try:
            if f.geomType() != "CYLINDER":
                continue
        except Exception:
            continue
        info = _cylinder_of(f)
        if info is None:
            continue
        radius, axis, _loc = info
        try:
            centre = f.Center()
            centre = [_round(centre.x, 3), _round(centre.y, 3), _round(centre.z, 3)]
            area = _round(f.Area(), 3)
        except Exception:
            centre, area = None, None
        key = f"r={radius}"
        group = cyl_groups.setdefault(key, {"radius": radius, "count": 0, "instances": []})
        group["count"] += 1
        if len(group["instances"]) < 8:
            group["instances"].append({"center": centre, "axis": axis, "area": area})
    report["cylindrical_faces"] = sorted(
        cyl_groups.values(), key=lambda g: (g["radius"] or 0)
    )[:max_items]

    # --- planar faces, largest first (the ones you build workplanes on) ------
    planes = []
    for f in faces:
        try:
            if f.geomType() != "PLANE":
                continue
            n = f.normalAt()
            c = f.Center()
            planes.append({
                "area": _round(f.Area(), 3),
                "center": [_round(c.x, 3), _round(c.y, 3), _round(c.z, 3)],
                "normal": [_round(n.x, 3), _round(n.y, 3), _round(n.z, 3)],
            })
        except Exception:
            continue
    planes.sort(key=lambda p: -(p["area"] or 0))
    report["planar_faces_largest"] = planes[:12]
    report["planar_face_count"] = len(planes)

    # --- circular edges grouped by radius (hole rims, existing fillets) ------
    circ = {}
    edge_types = {}
    total_edge_length = 0.0
    for e in edges:
        try:
            gt = e.geomType()
        except Exception:
            gt = "UNKNOWN"
        edge_types[gt] = edge_types.get(gt, 0) + 1
        try:
            total_edge_length += float(e.Length())
        except Exception:
            pass
        if gt in ("CIRCLE", "ARC"):
            try:
                r = _round(e.radius(), 4)
            except Exception:
                continue
            entry = circ.setdefault(r, {"radius": r, "count": 0, "centers": []})
            entry["count"] += 1
            try:
                c = e.Center()
                if len(entry["centers"]) < 6:
                    entry["centers"].append(
                        [_round(c.x, 3), _round(c.y, 3), _round(c.z, 3)]
                    )
            except Exception:
                pass
    report["edge_types"] = edge_types
    report["total_edge_length"] = _round(total_edge_length, 2)
    report["circular_edges"] = sorted(circ.values(), key=lambda g: g["radius"] or 0)[:max_items]

    return report


# ----------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------

def job_analyze(job):
    step_file = osp.expanduser(job["step_file"])
    shape = to_shape(cq.importers.importStep(step_file))
    return {"ok": True, "report": analyze_shape(shape)}


def job_render(job):
    step_file = osp.expanduser(job["step_file"])
    output_dir = osp.expanduser(job["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    views = job.get("views") or ["toprightiso"]
    prefix = job.get("prefix", "tmp")
    size = int(job.get("image_size", 800))

    shape = to_shape(cq.importers.importStep(step_file))
    images = _render_views(shape, output_dir, prefix, views, size, "png",
                           step_file=step_file)
    return {"ok": True, "images": images}


def _render_views(shape, output_dir, prefix, views, size, ext, step_file=None):
    """
    Fallback renderer for the benchmark views.

    Not `src.utils.cadquery_rendering`: its Windows path never fits the model,
    never sets a view-up (top/bottom render blank), and rotates the camera after
    positioning it. See brep_render.BENCHMARK_VIEWS.
    """
    from harness.cq import brep_render

    if step_file is None:
        # Fall back to writing the shape out so the renderer can load it.
        step_file = osp.join(output_dir, "_render_input.step")
        exporters.export(shape, step_file, exportType="STEP")

    return brep_render.render_benchmark_views(
        step_path=step_file, output_dir=output_dir, prefix=prefix,
        views=views, size=size, ext=ext,
    )


def job_execute(job):
    """Run `my_cad_function`, export STEP, render, analyse. Same contract as
    the organisers' `src/harnesses/cadquery_script.py`."""
    script_file = osp.expanduser(job["script_file"])
    output_dir = osp.expanduser(job["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    views = job.get("views") or ["toprightiso"]
    size = int(job.get("image_size", 800))

    args = {"output_dir": output_dir}
    if job.get("input_file"):
        args["input_file"] = osp.expanduser(job["input_file"])

    with open(script_file, "r", encoding="utf-8") as fh:
        source = fh.read()

    exec_globals = {
        "cq": cq,
        "cadquery": cq,
        "Workplane": cq.Workplane,
        "Assembly": cq.Assembly,
        "exporters": exporters,
        "os": os,
        "sys": sys,
        "__builtins__": __builtins__,
    }

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    result = None
    error = None

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(source, exec_globals)
            if "my_cad_function" not in exec_globals:
                raise ValueError(
                    "No function named 'my_cad_function' was defined by the script."
                )
            result = exec_globals["my_cad_function"](args)
    except BaseException as exc:  # noqa: BLE001 - report anything, including OCC errors
        error = f"{type(exc).__name__}: {exc}"
        stderr_buf.write("\n" + traceback.format_exc())

    payload = {
        "ok": False,
        "error": error,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "step_path": None,
        "images": [],
        "report": None,
    }

    if result is None:
        if payload["error"] is None:
            payload["error"] = "my_cad_function returned None (no shape produced)."
        return payload

    shape = to_shape(result)
    if shape is None:
        payload["error"] = "Could not extract a solid from the returned object."
        return payload

    step_path = osp.join(output_dir, "tmp.step")
    try:
        exporters.export(shape, step_path, exportType="STEP")
        payload["step_path"] = step_path if osp.exists(step_path) else None
    except Exception as exc:
        payload["error"] = f"STEP export failed: {exc}"
        return payload

    payload["report"] = analyze_shape(shape)
    # Same renderer as the deliverables.
    try:
        from harness.cq import vtk_render

        payload["images"] = vtk_render.render_benchmark_views(
            step_path=step_path, output_dir=output_dir, prefix="tmp",
            views=views, size=size, ext="png",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"VTK render unavailable ({exc}); using the mesh renderer",
              file=sys.stderr)
        payload["images"] = []

    if not payload["images"]:
        payload["images"] = _render_views(shape, output_dir, "tmp", views, size,
                                          "png", step_file=step_path)
    payload["ok"] = payload["step_path"] is not None
    return payload


def job_step_to_views(job):
    """Rendered views, with bbox overlay and labels matching step_to_json."""
    from harness.cq import brep_render

    step_file = osp.expanduser(job["step_file"])
    output_dir = osp.expanduser(job["output_dir"])
    views = job.get("views") or ["iso_top_right"]
    draw_bbox = bool(job.get("draw_bbox", True))
    label_faces = bool(job.get("label_faces", True))
    label_ids = job.get("label_ids")
    resolution = int(job.get("resolution", brep_render.DEFAULT_RESOLUTION_PX))
    prefix = job.get("prefix", "view")

    # Kernel-quality render with overlays composited on top.
    result, backend = None, "vtk"
    try:
        from harness.cq import vtk_render

        result = vtk_render.render_annotated_views(
            step_path=step_file, output_dir=output_dir, prefix=prefix,
            views=views, size=resolution, draw_bbox=draw_bbox,
            label_faces=label_faces, label_ids=label_ids,
            view_table=brep_render.ALL_VIEWS,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"VTK render unavailable ({exc}); using the mesh renderer",
              file=sys.stderr)

    if not (result and result.get("images")):
        backend = "matplotlib"
        result = brep_render.render_views(
            step_path=step_file, output_dir=output_dir, views=views,
            draw_bbox=draw_bbox, label_faces=label_faces,
            draw_edges=bool(job.get("draw_edges", False)),
            resolution=resolution, prefix=prefix, label_ids=label_ids,
        )

    result["render_backend"] = backend
    result["ok"] = bool(result.get("images"))
    if not result["ok"]:
        result["error"] = "no images were produced"
    return result


def job_step_to_json(job):
    """Full B-rep JSON to disk, token-budgeted digest returned."""
    from harness.cq import brep_extract
    from harness.reporting import format_brep_digest

    step_file = osp.expanduser(job["step_file"])
    brep = brep_extract.extract_brep(
        step_file,
        include_edges=bool(job.get("include_edges", True)),
        include_adjacency=bool(job.get("include_adjacency", True)),
    )

    json_path = job.get("json_path")
    if json_path:
        json_path = osp.expanduser(json_path)
        os.makedirs(osp.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(brep, fh, indent=2, default=str)

    digest = format_brep_digest(
        brep,
        max_chars=int(job.get("max_chars", 6000)),
        include_edges=bool(job.get("include_edges", True)),
        face_type=job.get("face_type"),
        min_area=job.get("min_area"),
        max_area=job.get("max_area"),
        min_radius=job.get("min_radius"),
        max_radius=job.get("max_radius"),
        ids=job.get("ids"),
        touching_only=job.get("touching_only"),
        offset=job.get("offset", 0),
    )

    return {
        "ok": True,
        "json_path": json_path,
        "digest": digest,
        "num_solids": brep["num_solids"],
        "multi_solid": brep["multi_solid"],
        "total_faces": brep["total_faces"],
        "total_edges": brep["total_edges"],
        "part_bounding_box": brep["part_bounding_box"],
    }


def job_convert(job):
    """Emit the .stl and 7 views beside a final STEP."""
    step_file = osp.expanduser(job["step_file"])
    views = job.get("views") or DEFAULT_VIEWS
    ext = job.get("image_ext", "png")
    size = int(job.get("image_size", 1024))

    base = osp.splitext(step_file)[0]
    out_dir = osp.dirname(step_file)
    prefix = osp.basename(base)

    shape = to_shape(cq.importers.importStep(step_file))
    if shape is None:
        return {"ok": False, "error": "could not import STEP"}

    stl_path = base + ".stl"
    try:
        exporters.export(shape, stl_path, exporters.ExportTypes.STL)
    except Exception as exc:
        return {"ok": False, "error": f"STL export failed: {exc}"}

    # Deliverables go through the kernel's VTK pipeline for smooth shading.
    images, render_backend = [], "vtk"
    try:
        from harness.cq import vtk_render

        images = vtk_render.render_benchmark_views(
            step_path=step_file, output_dir=out_dir, prefix=prefix,
            views=views, size=size, ext=ext,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"VTK render unavailable ({exc}); falling back to the mesh renderer",
              file=sys.stderr)

    if not images:
        render_backend = "matplotlib"
        images = _render_views(shape, out_dir, prefix, views, size, ext,
                               step_file=step_file)

    return {
        "ok": osp.exists(stl_path),
        "stl": stl_path if osp.exists(stl_path) else None,
        "images": images,
        "render_backend": render_backend,
    }


JOBS = {
    "analyze": job_analyze,
    "execute": job_execute,
    "render": job_render,
    "convert": job_convert,
    "step_to_views": job_step_to_views,
    "step_to_json": job_step_to_json,
}


def run_job(job: dict) -> dict:
    try:
        handler = JOBS[job["kind"]]
    except KeyError:
        return {"ok": False, "error": f"unknown job kind {job.get('kind')!r}"}
    try:
        return handler(job)
    except BaseException as exc:  # noqa: BLE001 - report anything, including OCC errors
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stderr": traceback.format_exc(),
        }


def _execute_to_file(job_path: str, result_path: str) -> None:
    with open(job_path, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    result = run_job(job)
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)


def serve() -> None:
    """
    Long-lived mode: import CadQuery once, then handle jobs until told to stop.

    Protocol, one line per exchange on stdin: "<job_path>|<result_path>";
    reply "OK" or "ERR <message>" on stdout; "EXIT" stops. stdout is reserved
    for the protocol, so job output is redirected to stderr.
    """
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line or line == "EXIT":
            break
        try:
            job_path, result_path = line.split("|", 1)
            with redirect_stdout(sys.stderr):
                _execute_to_file(job_path, result_path)
            sys.stdout.write("OK\n")
        except BaseException as exc:  # noqa: BLE001
            sys.stdout.write(f"ERR {type(exc).__name__}: {exc}\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", help="path to the job JSON (one-shot mode)")
    parser.add_argument("--result", help="path to write the result JSON")
    parser.add_argument("--serve", action="store_true",
                        help="stay alive and handle jobs from stdin")
    ns = parser.parse_args()

    if ns.serve:
        serve()
        return

    if not ns.job or not ns.result:
        parser.error("--job and --result are required unless --serve is given")

    _execute_to_file(ns.job, ns.result)


if __name__ == "__main__":
    main()
