"""
Turn geometry reports into compact text for the model.

Token efficiency is a scored criterion, so these formatters deliberately print
the numbers an engineer would write down and drop everything else.
"""

from typing import Optional


def _fmt_vec(v, nd=3):
    if not v:
        return "?"
    return "(" + ", ".join(f"{x:g}" if isinstance(x, (int, float)) else "?" for x in v) + ")"


def format_report(report: dict, title: str = "Geometry", detail: str = "full") -> str:
    """Render an `analyze_shape` report as text. detail: 'full' | 'brief'."""
    if not report:
        return f"{title}: unavailable"
    if report.get("error"):
        return f"{title}: ERROR {report['error']}"

    lines = [f"## {title}"]

    bb = report.get("bounding_box") or {}
    if bb.get("size"):
        lines.append(
            f"bbox size {_fmt_vec(bb['size'])} mm, centre {_fmt_vec(bb.get('center'))}, "
            f"diag {bb.get('diagonal')}"
        )
        lines.append(
            f"extents  x[{bb.get('xmin')}, {bb.get('xmax')}]  "
            f"y[{bb.get('ymin')}, {bb.get('ymax')}]  z[{bb.get('zmin')}, {bb.get('zmax')}]"
        )
    lines.append(
        f"volume {report.get('volume')} mm^3, surface area {report.get('surface_area')} mm^2, "
        f"valid={report.get('is_valid')}"
    )

    counts = report.get("counts") or {}
    if counts and "error" not in counts:
        lines.append(
            "topology: " + ", ".join(f"{k}={v}" for k, v in counts.items())
        )
    if report.get("face_types"):
        lines.append(
            "face types: " + ", ".join(f"{k}={v}" for k, v in report["face_types"].items())
        )

    if detail == "brief":
        return "\n".join(lines)

    cyls = report.get("cylindrical_faces") or []
    if cyls:
        lines.append("cylindrical faces (holes / bosses / fillet surfaces):")
        for group in cyls[:14]:
            first = group["instances"][0] if group.get("instances") else {}
            lines.append(
                f"  r={group['radius']}  x{group['count']}  "
                f"axis {_fmt_vec(first.get('axis'))}  centre {_fmt_vec(first.get('center'))}"
            )
        if len(cyls) > 14:
            lines.append(f"  ... {len(cyls) - 14} more radius groups")

    circles = report.get("circular_edges") or []
    if circles:
        lines.append(
            f"circular edges by radius ({report.get('planar_face_count', '?')} planar faces total):"
        )
        for group in circles[:12]:
            lines.append(f"  r={group['radius']}  x{group['count']}")
        if len(circles) > 12:
            lines.append(f"  ... {len(circles) - 12} more radius groups")

    planes = report.get("planar_faces_largest") or []
    if planes:
        lines.append("largest planar faces (good workplane candidates):")
        for p in planes[:6]:
            lines.append(
                f"  area={p['area']}  centre {_fmt_vec(p['center'])}  normal {_fmt_vec(p['normal'])}"
            )

    return "\n".join(lines)


def diff_reports(before: dict, after: dict) -> dict:
    """Before/after delta. Pure dict arithmetic, no CAD dependency."""
    if not before or not after:
        return {}

    def _r(x, nd=4):
        try:
            return round(float(x), nd)
        except Exception:
            return None

    out = {}

    for key in ("volume", "surface_area"):
        b, a = before.get(key), after.get(key)
        if b and a:
            out[key] = {
                "before": b,
                "after": a,
                "delta": _r(a - b),
                "pct": _r(100.0 * (a - b) / b, 2),
            }

    bb_b, bb_a = before.get("bounding_box") or {}, after.get("bounding_box") or {}
    if bb_b.get("size") and bb_a.get("size"):
        out["bbox_size"] = {
            "before": bb_b["size"],
            "after": bb_a["size"],
            "delta": [_r((a or 0) - (b or 0), 3)
                      for b, a in zip(bb_b["size"], bb_a["size"])],
        }
    if bb_b.get("center") and bb_a.get("center"):
        out["bbox_center_shift"] = [
            _r((a or 0) - (b or 0), 3)
            for b, a in zip(bb_b["center"], bb_a["center"])
        ]

    cb, ca = before.get("counts") or {}, after.get("counts") or {}
    if cb and ca and "error" not in cb and "error" not in ca:
        out["counts"] = {
            k: {"before": cb.get(k), "after": ca.get(k),
                "delta": (ca.get(k) or 0) - (cb.get(k) or 0)}
            for k in ("solids", "faces", "edges", "vertices")
        }

    # "identical" == the edit did nothing; the agent must be told loudly.
    out["appears_unchanged"] = bool(
        before.get("volume") == after.get("volume")
        and cb.get("faces") == ca.get("faces")
        and cb.get("edges") == ca.get("edges")
    )
    return out


def format_diff(diff: dict) -> str:
    """Render a before/after diff. This is the agent's objective feedback."""
    if not diff:
        return "Diff vs original: unavailable."

    lines = ["## Change vs the ORIGINAL part"]

    vol = diff.get("volume")
    if vol:
        lines.append(
            f"volume {vol['before']} -> {vol['after']}  ({vol['delta']:+g} mm^3, {vol['pct']:+g}%)"
        )
    area = diff.get("surface_area")
    if area:
        lines.append(
            f"surface area {area['before']} -> {area['after']}  ({area['pct']:+g}%)"
        )
    bbox = diff.get("bbox_size")
    if bbox:
        lines.append(
            f"bbox size {_fmt_vec(bbox['before'])} -> {_fmt_vec(bbox['after'])}  "
            f"delta {_fmt_vec(bbox['delta'])}"
        )
    shift = diff.get("bbox_center_shift")
    if shift and any(abs(x or 0) > 1e-6 for x in shift):
        lines.append(f"WARNING: part centre moved by {_fmt_vec(shift)} - did you mean to move it?")

    counts = diff.get("counts")
    if counts:
        lines.append(
            "topology delta: "
            + ", ".join(f"{k} {v['before']}->{v['after']} ({v['delta']:+d})"
                        for k, v in counts.items())
        )

    if diff.get("appears_unchanged"):
        lines.append("!! The output is geometrically IDENTICAL to the input. The edit did nothing.")

    return "\n".join(lines)


def _fmt_num(x, nd=3):
    if x is None:
        return "?"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _face_line(face: dict) -> str:
    """One line per face, with the numbers you would type into a script."""
    parts = [f"{face['id']} {face.get('type', '?')}"]
    parts.append(f"area={_fmt_num(face.get('area'))}")
    parts.append(f"c={_fmt_vec(face.get('centroid'))}")
    if face.get("normal"):
        parts.append(f"n={_fmt_vec(face['normal'])}")

    params = face.get("geometry_params") or {}
    if "radius" in params:
        parts.append(f"r={_fmt_num(params['radius'])}")
    if "major_radius" in params:
        parts.append(f"R={_fmt_num(params['major_radius'])}/{_fmt_num(params.get('minor_radius'))}")
    if "axis_direction" in params:
        parts.append(f"axis={_fmt_vec(params['axis_direction'])}")
    if face.get("touches_bbox"):
        parts.append("[on bbox]")
    return "  " + "  ".join(parts)


def filter_faces(faces: list, face_type=None, min_area=None, max_area=None,
                 min_radius=None, max_radius=None, ids=None,
                 touching_only=False) -> list:
    """Narrow a face list by the criteria `step_to_json` exposes to the agent."""
    if ids:
        wanted = {str(i).upper() for i in ids}
        return [f for f in faces if str(f.get("id", "")).upper() in wanted]

    out = []
    for f in faces:
        if face_type and str(f.get("type", "")).upper() != str(face_type).upper():
            continue
        if touching_only and not f.get("touches_bbox"):
            continue
        area = f.get("area")
        if min_area is not None and (area is None or area < float(min_area)):
            continue
        if max_area is not None and (area is None or area > float(max_area)):
            continue
        if min_radius is not None or max_radius is not None:
            r = (f.get("geometry_params") or {}).get("radius")
            if r is None:
                continue
            if min_radius is not None and r < float(min_radius):
                continue
            if max_radius is not None and r > float(max_radius):
                continue
        out.append(f)
    return out


def _complete_face_survey(faces: list) -> list:
    """
    Complete face-type and radius census, never truncated - so a cut per-face
    listing cannot hide a feature.
    """
    lines = []

    types = {}
    for f in faces:
        types[f.get("type", "?")] = types.get(f.get("type", "?"), 0) + 1
    if types:
        lines.append("face types (ALL faces): "
                     + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))

    radii = {}
    for f in faces:
        r = (f.get("geometry_params") or {}).get("radius")
        if r is None:
            continue
        entry = radii.setdefault(round(float(r), 4), {"count": 0, "ids": []})
        entry["count"] += 1
        if len(entry["ids"]) < 3:
            entry["ids"].append(f.get("id"))

    if radii:
        lines.append(f"every cylindrical/spherical radius present "
                     f"({len(radii)} distinct):")
        for r in sorted(radii):
            entry = radii[r]
            lines.append(f"  r={_fmt_num(r)}  x{entry['count']}  "
                         f"e.g. {', '.join(str(i) for i in entry['ids'])}")

    return lines


def format_brep_digest(brep: dict, max_chars: int = 6000,
                       include_edges: bool = True, **face_filters) -> str:
    """
    `extract_brep` -> text the model can afford (3.4 MB -> ~5 KB on the heaviest
    part). Faces ranked bbox-touching first, then by area; what is dropped is
    stated so the model knows to filter rather than assume.
    """
    if not brep:
        return "B-rep data unavailable."

    lines = ["## B-rep structure"]
    lines.append(
        f"{brep.get('num_solids')} solid body(ies), "
        f"{brep.get('total_faces')} faces, {brep.get('total_edges')} edges, "
        f"total volume {_fmt_num(brep.get('total_volume'), 6)} mm^3"
    )
    lines.append(f"face/edge label convention: {brep.get('label_convention')}")

    bbox = brep.get("part_bounding_box") or {}
    if bbox:
        lines.append(
            f"part bbox min {_fmt_vec(bbox.get('min'))} max {_fmt_vec(bbox.get('max'))} "
            f"size {_fmt_vec(brep.get('part_size'))}"
        )

    solids = brep.get("solids") or []
    if len(solids) > 1:
        lines.append("\nbodies:")
        for s in solids[:12]:
            lines.append(
                f"  {s['solid_id']}: {s['num_faces']} faces, "
                f"vol={_fmt_num(s.get('volume'))}, "
                f"centroid={_fmt_vec(s.get('centroid'))}, "
                f"size={_fmt_vec([round(s['bounding_box']['max'][k] - s['bounding_box']['min'][k], 3) for k in range(3)])}"
            )
        if len(solids) > 12:
            lines.append(f"  ... {len(solids) - 12} more bodies")

    # --- complete survey: never truncated, so nothing is silently hidden ----
    all_faces = [f for s in solids for f in s.get("faces", [])]
    survey = _complete_face_survey(all_faces)
    if survey:
        lines.append("")
        lines.extend(survey)

    # --- faces, most useful first ------------------------------------------
    offset = int(face_filters.pop("offset", 0) or 0)
    selected = filter_faces(all_faces, **{k: v for k, v in face_filters.items()
                                          if v is not None})
    filtered = len(selected) != len(all_faces)

    ranked = sorted(
        selected,
        key=lambda f: (0 if f.get("touches_bbox") else 1, -(f.get("area") or 0)),
    )
    if offset:
        ranked = ranked[offset:]

    header_len = sum(len(line) + 1 for line in lines)
    budget = max(max_chars - header_len, 500)

    scope = (f"{len(selected)} matching this filter, of {len(all_faces)} total"
             if filtered else f"{len(all_faces)} total")
    if offset:
        scope += f", skipping the first {offset}"
    lines.append(f"\nfaces ({scope}, bbox-touching listed first):")

    used = 0
    shown = 0
    for face in ranked:
        line = _face_line(face)
        if used + len(line) > budget * (0.75 if include_edges else 1.0):
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1

    remaining = len(ranked) - shown
    if remaining > 0:
        lines.append(
            f"  ... {remaining} more not listed. Every radius in the part is "
            f"already summarised above; to see specific faces call step_to_json "
            f"with face_type / min_radius / max_radius / min_area / ids, or "
            f"offset={offset + shown} to continue this list."
        )
    elif not ranked:
        lines.append("  (nothing matched - widen the filter)")

    # --- edges, grouped ------------------------------------------------------
    if include_edges:
        all_edges = [e for s in solids for e in s.get("edges", [])]
        if all_edges:
            by_radius = {}
            type_counts = {}
            for e in all_edges:
                type_counts[e.get("type", "?")] = type_counts.get(e.get("type", "?"), 0) + 1
                r = (e.get("geometry_params") or {}).get("radius")
                if r is not None:
                    entry = by_radius.setdefault(r, {"radius": r, "count": 0, "ids": []})
                    entry["count"] += 1
                    if len(entry["ids"]) < 4:
                        entry["ids"].append(e["id"])

            lines.append(
                f"\nedges ({len(all_edges)} total): "
                + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
            )
            if by_radius:
                lines.append("circular edges grouped by radius:")
                for group in sorted(by_radius.values(), key=lambda g: g["radius"])[:20]:
                    lines.append(
                        f"  r={_fmt_num(group['radius'])}  x{group['count']}  "
                        f"e.g. {', '.join(group['ids'])}"
                    )
                if len(by_radius) > 20:
                    lines.append(f"  ... {len(by_radius) - 20} more radius groups")

    return "\n".join(lines)


def truncate(text: Optional[str], limit: int = 2500, tail: int = 600) -> str:
    """Trim the middle - tracebacks put the useful line at the end."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit - tail
    return (
        text[:head]
        + f"\n... [{len(text) - limit} characters trimmed] ...\n"
        + text[-tail:]
    )
