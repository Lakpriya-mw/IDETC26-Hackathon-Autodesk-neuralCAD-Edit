"""Measurement tools, so the agent never has to invent a dimension."""

from harness.cq import client as cq_client
from harness.reporting import format_report
from harness.tools.base import ToolContext, ToolResult, tool


def _report_for(ctx: ToolContext, target: str):
    """Resolve 'input' / 'current' to (step_path, report, label)."""
    target = (target or "input").lower()
    if target in ("current", "output", "last", "result"):
        if not ctx.last_output_step:
            return None, None, "no build has succeeded yet"
        if ctx.last_report is None:
            res = cq_client.analyze(ctx.last_output_step)
            ctx.last_report = res.get("report") if res.get("ok") else None
        return ctx.last_output_step, ctx.last_report, "your latest build"
    if ctx.input_report is None:
        res = cq_client.analyze(ctx.input_step)
        ctx.input_report = res.get("report") if res.get("ok") else None
    return ctx.input_step, ctx.input_report, "the original part"


@tool(
    name="inspect_geometry",
    description=(
        "Full measured description of a part: bounding box, volume, surface area, "
        "face/edge/solid counts, face-type histogram, every cylindrical face grouped "
        "by radius with its axis and centre, circular edges grouped by radius, and "
        "the largest planar faces. Call this FIRST, before writing any geometry."
    ),
    params={
        "target": "'input' (the original part, default) or 'current' (your latest build)",
    },
)
def inspect_geometry(ctx: ToolContext, target: str = "input"):
    _path, report, label = _report_for(ctx, target)
    if report is None:
        return ToolResult(text=f"Could not analyse {label}.", ok=False)
    return ToolResult(text=format_report(report, title=f"Geometry of {label}"))


@tool(
    name="query_entities",
    description=(
        "Filter the measured entity list to find a specific feature. Use it to turn "
        "a phrase in the request ('the big fillet', 'the two mounting holes', 'the flat "
        "bottom') into exact radii, centres, axes and areas you can type into a script."
    ),
    params={
        "kind": "'cylinders' | 'circles' | 'planes' - which entity list to search",
        "min_radius": "float, optional lower bound on radius",
        "max_radius": "float, optional upper bound on radius",
        "axis": "optional 'x' | 'y' | 'z' - keep only entities whose axis/normal is along it",
        "target": "'input' (default) or 'current'",
        "limit": "int, max rows to return (default 20)",
    },
)
def query_entities(ctx: ToolContext, kind: str = "cylinders", min_radius=None,
                   max_radius=None, axis: str = None, target: str = "input",
                   limit: int = 20):
    _path, report, label = _report_for(ctx, target)
    if report is None:
        return ToolResult(text=f"Could not analyse {label}.", ok=False)

    kind = (kind or "cylinders").lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get((axis or "").lower())

    def radius_ok(r):
        if r is None:
            return False
        if min_radius is not None and r < float(min_radius):
            return False
        if max_radius is not None and r > float(max_radius):
            return False
        return True

    def axis_ok(vec):
        if axis_index is None:
            return True
        if not vec:
            return False
        return abs(vec[axis_index] or 0) > 0.9

    rows = []
    if kind.startswith("cyl"):
        for group in report.get("cylindrical_faces", []):
            if not radius_ok(group.get("radius")):
                continue
            for inst in group.get("instances", []):
                if not axis_ok(inst.get("axis")):
                    continue
                rows.append(
                    f"cylinder r={group['radius']}  centre={inst.get('center')}  "
                    f"axis={inst.get('axis')}  area={inst.get('area')}"
                )
    elif kind.startswith("circ"):
        for group in report.get("circular_edges", []):
            if not radius_ok(group.get("radius")):
                continue
            rows.append(
                f"circular edge r={group['radius']}  count={group['count']}  "
                f"centres={group.get('centers')}"
            )
    elif kind.startswith("plan"):
        for p in report.get("planar_faces_largest", []):
            if not axis_ok(p.get("normal")):
                continue
            rows.append(
                f"plane area={p['area']}  centre={p['center']}  normal={p['normal']}"
            )
    else:
        return ToolResult(
            text="ERROR: kind must be one of 'cylinders', 'circles', 'planes'.", ok=False
        )

    if not rows:
        return ToolResult(
            text=(
                f"No {kind} in {label} matched that filter. "
                f"Widen the radius range or drop the axis constraint, or call "
                f"inspect_geometry to see what is actually there."
            )
        )

    limit = int(limit or 20)
    body = "\n".join(rows[:limit])
    extra = f"\n... {len(rows) - limit} more matches" if len(rows) > limit else ""
    return ToolResult(text=f"{len(rows)} match(es) in {label}:\n{body}{extra}")


@tool(
    name="compare_to_original",
    description=(
        "Numeric diff of your latest build against the original part: volume, surface "
        "area, bounding box, centre shift, and face/edge counts. Use it to confirm the "
        "edit did what you intended and nothing more."
    ),
    params={},
)
def compare_to_original(ctx: ToolContext):
    from harness.reporting import diff_reports, format_diff

    if not ctx.last_output_step:
        return ToolResult(text="No successful build yet - nothing to compare.", ok=False)

    _p, before, _l = _report_for(ctx, "input")
    _p2, after, _l2 = _report_for(ctx, "current")
    if not before or not after:
        return ToolResult(text="Could not analyse one of the two shapes.", ok=False)
    return ToolResult(text=format_diff(diff_reports(before, after)))
