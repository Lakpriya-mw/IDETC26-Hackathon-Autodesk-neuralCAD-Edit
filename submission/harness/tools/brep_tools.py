"""
`step_to_views` and `step_to_json`, designed as a pair: both use OpenCascade's
deterministic traversal order, so a face tagged F12 in an image is the F12 whose
radius the JSON reports.

Both run in the sandboxed CadQuery worker.
"""

import os.path as osp

from harness.cq import client as cq_client
from harness.cq.brep_render import VIEW_NAMES
from harness.tools.base import ToolContext, ToolResult, tool

MAX_VIEWS_PER_CALL = 3


def _resolve_target(ctx: ToolContext, target: str):
    """'input' -> the original part, 'current' -> the newest successful build."""
    target = (target or "current").lower()
    if target.startswith(("in", "orig", "start")):
        return ctx.input_step, "the ORIGINAL part", "orig"
    if not ctx.last_output_step:
        return None, "no build has succeeded yet", None
    return ctx.last_output_step, "your latest build", "cur"


@tool(
    name="step_to_views",
    description=(
        "Render a part from named viewpoints and look at the images. Each view can "
        "carry the projected 3D bounding box with real corner coordinates, and a "
        "label on every face that touches that bounding box - the labels are the "
        "same F#/S#F# ids that step_to_json reports, so you can match a face in the "
        "picture to its exact radius/axis/centroid in the JSON. "
        "Views: " + ", ".join(VIEW_NAMES) + ". "
        "Rendering is the slowest thing you can do, so ask for the fewest views that "
        "settle your question - usually one, at most two. Prefer an isometric unless "
        "you need a true orthographic silhouette."
    ),
    params={
        "views": f"list of view names (max {MAX_VIEWS_PER_CALL} per call)",
        "target": "'current' (your latest build, default) or 'input' (the original part)",
        "label_ids": "list of face ids to tag, e.g. ['F142','F143'] - tags exactly "
                     "these instead of the outer envelope. The way to locate a "
                     "feature in the MIDDLE of a part, which touches no bounding-box "
                     "plane and is otherwise never labelled.",
        "draw_bbox": "bool, overlay the bounding box + corner coordinates (default true)",
        "label_faces": "bool, tag bbox-touching faces with their F# ids (default true; "
                       "ignored when label_ids is given)",
    },
    expensive=True,
)
def step_to_views(ctx: ToolContext, views=None, target: str = "current",
                  draw_bbox: bool = True, label_faces: bool = True,
                  label_ids=None):
    step_file, label, prefix = _resolve_target(ctx, target)
    if step_file is None:
        return ToolResult(
            text=f"{label} - render the input instead with target='input'.", ok=False
        )

    if isinstance(views, str):
        views = [views]
    views = list(views or ["iso_top_right"])

    unknown = [v for v in views if v not in VIEW_NAMES]
    if unknown:
        return ToolResult(
            text=(f"ERROR: unknown view(s) {unknown}. "
                  f"Valid views: {', '.join(VIEW_NAMES)}"),
            ok=False,
        )

    trimmed = ""
    if len(views) > MAX_VIEWS_PER_CALL:
        trimmed = (f" (you asked for {len(views)}; rendering the first "
                   f"{MAX_VIEWS_PER_CALL} - rendering is expensive)")
        views = views[:MAX_VIEWS_PER_CALL]

    if isinstance(label_ids, str):
        label_ids = [label_ids]

    out_dir = osp.join(ctx.work_dir, "views")
    settings = ctx.extras.get("view_settings", {})

    result = cq_client.step_to_views(
        step_file=step_file,
        output_dir=out_dir,
        views=views,
        draw_bbox=bool(draw_bbox),
        label_faces=bool(label_faces),
        draw_edges=settings.get("draw_edges", False),
        resolution=settings.get("resolution", 900),
        prefix=f"{prefix}_{ctx.extras.get('view_counter', 0)}",
        label_ids=label_ids,
    )
    ctx.extras["view_counter"] = ctx.extras.get("view_counter", 0) + 1

    images = [p for p in (result.get("images") or []) if osp.exists(p)]
    if not images:
        return ToolResult(
            text=f"Rendering failed: {result.get('error', 'no images produced')}",
            ok=False,
        )

    note = f"Views of {label}: {', '.join(views)}{trimmed}."
    if label_ids:
        note += (f" Tagged {result.get('labelled_faces', 0)} requested face(s) "
                 f"in orange.")
        missing = result.get("missing_ids") or []
        if missing:
            note += (f" These ids do not exist in this part and were skipped: "
                     f"{', '.join(missing)}.")
    if result.get("multi_solid"):
        note += (f" This part has {result['num_solids']} separate bodies, all drawn "
                 f"together; face labels are prefixed S<body>.")
    return ToolResult(text=note, images=images)


@tool(
    name="step_to_json",
    description=(
        "Extract the B-rep structure of a part: per-face type, area, centroid, normal "
        "and analytic parameters (cylinder radius/axis, cone angle, torus radii, plane "
        "normal), plus edges grouped by radius. Face ids match the labels drawn by "
        "step_to_views. "
        "The reply always begins with a COMPLETE survey - every face type and every "
        "distinct radius in the part - and then lists individual faces, which on a "
        "large part is necessarily a subset. Narrow it with face_type / min_radius / "
        "max_radius / min_area / ids / touching_only, or page with offset. "
        "Use it to turn a phrase in the request ('the hole', 'the largest fillet') "
        "into exact numbers, and to confirm after a build that the feature you "
        "intended actually exists at the size you meant."
    ),
    params={
        "target": "'current' (your latest build, default) or 'input' (the original part)",
        "face_type": "'CYLINDER' | 'PLANE' | 'CONE' | 'TORUS' | 'SPHERE' | 'BSPLINE' - list only this type",
        "min_radius": "float, only faces with radius >= this (finds holes/bosses of a size)",
        "max_radius": "float, only faces with radius <= this",
        "min_area": "float, only faces at least this big",
        "ids": "list of face ids to show in full, e.g. ['F12','F13']",
        "touching_only": "bool, only faces on the bounding box (the outer envelope)",
        "offset": "int, skip this many ranked faces - use it to page through a long list",
        "include_edges": "bool, include edge detail (default true; set false on huge parts)",
        "detail": "int, roughly how many characters to return (1000-20000, default 6000)",
    },
)
def step_to_json(ctx: ToolContext, target: str = "current",
                 include_edges: bool = True, detail: int = 6000,
                 face_type: str = None, min_radius=None, max_radius=None,
                 min_area=None, ids=None, touching_only: bool = False,
                 offset: int = 0):
    step_file, label, _prefix = _resolve_target(ctx, target)
    if step_file is None:
        return ToolResult(
            text=f"{label} - inspect the input instead with target='input'.", ok=False
        )

    try:
        detail = max(1000, min(int(detail), 20000))
    except (TypeError, ValueError):
        detail = 6000

    if isinstance(ids, str):
        ids = [ids]

    json_path = osp.join(ctx.work_dir, "brep_json",
                         f"{osp.splitext(osp.basename(step_file))[0]}_brep.json")

    result = cq_client.step_to_json(
        step_file=step_file,
        json_path=json_path,
        include_edges=bool(include_edges),
        max_chars=detail,
        face_type=face_type,
        min_radius=min_radius,
        max_radius=max_radius,
        min_area=min_area,
        ids=ids,
        touching_only=touching_only or None,
        offset=offset or None,
    )

    if not result.get("ok"):
        return ToolResult(
            text=f"B-rep extraction failed: {result.get('error')}", ok=False
        )

    return ToolResult(text=f"B-rep of {label}:\n{result['digest']}")
