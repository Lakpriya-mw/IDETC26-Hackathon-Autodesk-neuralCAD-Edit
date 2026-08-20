"""
B-rep -> JSON extraction.

Adapted from the user-supplied `step_to_json_and_views.py` (PART 1), keeping
its output schema and its F#/E# naming convention, with three changes forced
by this dataset:

1. **Fast edge->face adjacency.** The original compares every face-edge against
   every solid-edge with `IsSame` — O(faces x edges). The heaviest part here has
   1482 faces / 3749 edges, so that is millions of comparisons per call. This
   uses OCCT's own `TopTools_IndexedMapOfShape`, which is a hash lookup, and
   falls back to the original loop only if that import fails.

2. **Multi-solid labelling.** 24 of the 48 parts have more than one body (up to
   56). Bare `F3` is ambiguous across bodies, so labels become `S0F3` whenever
   the part has more than one solid, and stay `F3` when it has exactly one.
   The renderer uses the identical rule, which is what keeps the JSON and the
   images cross-referenceable.

3. **Optional edge detail.** Edges dominate the payload and are rarely needed in
   full, so callers can ask for faces only.
"""

import cadquery as cq

DEFAULT_NDIGITS = 6


# ---------------------------------------------------------------------------
# small helpers (unchanged in spirit from the source script)
# ---------------------------------------------------------------------------

def _vec_to_list(v, ndigits=DEFAULT_NDIGITS):
    return [round(float(v.x), ndigits), round(float(v.y), ndigits), round(float(v.z), ndigits)]


def _gp_dir_to_list(d, ndigits=DEFAULT_NDIGITS):
    return [round(d.X(), ndigits), round(d.Y(), ndigits), round(d.Z(), ndigits)]


def _gp_pnt_to_list(p, ndigits=DEFAULT_NDIGITS):
    return [round(p.X(), ndigits), round(p.Y(), ndigits), round(p.Z(), ndigits)]


def load_solids(step_path):
    """Every solid body in the STEP file."""
    result = cq.importers.importStep(step_path)
    solids = result.val().Solids()
    if not solids:
        raise ValueError(f"No solid bodies found in {step_path}")
    return solids


def face_label(solid_index, face_index, multi_solid):
    """`F3` for a single-body part, `S0F3` when several bodies exist."""
    return f"S{solid_index}F{face_index}" if multi_solid else f"F{face_index}"


def edge_label(solid_index, edge_index, multi_solid):
    return f"S{solid_index}E{edge_index}" if multi_solid else f"E{edge_index}"


# ---------------------------------------------------------------------------
# analytic parameters
# ---------------------------------------------------------------------------

def face_analytic_params(face):
    """Exact defining parameters for analytic surface types."""
    geom_type = face.geomType()
    try:
        surf = face._geomAdaptor()
        if geom_type == "CYLINDER":
            cyl = surf.Cylinder()
            return {
                "radius": round(cyl.Radius(), DEFAULT_NDIGITS),
                "axis_direction": _gp_dir_to_list(cyl.Axis().Direction()),
                "axis_location": _gp_pnt_to_list(cyl.Location()),
            }
        if geom_type == "CONE":
            cone = surf.Cone()
            return {
                "reference_radius": round(cone.RefRadius(), DEFAULT_NDIGITS),
                "semi_angle_rad": round(cone.SemiAngle(), DEFAULT_NDIGITS),
                "apex": _gp_pnt_to_list(cone.Apex()),
                "axis_direction": _gp_dir_to_list(cone.Axis().Direction()),
            }
        if geom_type == "SPHERE":
            sph = surf.Sphere()
            return {
                "radius": round(sph.Radius(), DEFAULT_NDIGITS),
                "center": _gp_pnt_to_list(sph.Location()),
            }
        if geom_type == "TORUS":
            torus = surf.Torus()
            return {
                "major_radius": round(torus.MajorRadius(), DEFAULT_NDIGITS),
                "minor_radius": round(torus.MinorRadius(), DEFAULT_NDIGITS),
                "center": _gp_pnt_to_list(torus.Location()),
                "axis_direction": _gp_dir_to_list(torus.Axis().Direction()),
            }
        if geom_type == "PLANE":
            pln = surf.Plane()
            return {
                "point_on_plane": _gp_pnt_to_list(pln.Location()),
                "plane_normal": _gp_dir_to_list(pln.Axis().Direction()),
            }
    except Exception:
        return {}
    return {}


def edge_analytic_params(edge):
    """Radius + axis for circular edges."""
    if edge.geomType() != "CIRCLE":
        return {}
    try:
        circ = edge._geomAdaptor().Circle()
        return {
            "radius": round(circ.Radius(), DEFAULT_NDIGITS),
            "axis_direction": _gp_dir_to_list(circ.Axis().Direction()),
            "center": _gp_pnt_to_list(circ.Location()),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# bounding boxes
# ---------------------------------------------------------------------------

def solid_bbox_dict(solid):
    bb = solid.BoundingBox()
    return {
        "min": [round(bb.xmin, DEFAULT_NDIGITS), round(bb.ymin, DEFAULT_NDIGITS),
                round(bb.zmin, DEFAULT_NDIGITS)],
        "max": [round(bb.xmax, DEFAULT_NDIGITS), round(bb.ymax, DEFAULT_NDIGITS),
                round(bb.zmax, DEFAULT_NDIGITS)],
    }


def union_bbox(bboxes):
    """Envelope of several bounding boxes - the whole part, not one body."""
    return {
        "min": [min(b["min"][k] for b in bboxes) for k in range(3)],
        "max": [max(b["max"][k] for b in bboxes) for k in range(3)],
    }


def compute_bbox_touching_faces(faces, bbox, tol=None):
    """
    True for each face lying on one of the 6 bounding planes.

    Call BEFORE tessellating: afterwards face.BoundingBox() is mesh-derived and
    the error can flip this test for a face genuinely on the boundary.
    """
    bmin, bmax = bbox["min"], bbox["max"]
    if tol is None:
        diag = sum((bmax[k] - bmin[k]) ** 2 for k in range(3)) ** 0.5
        tol = max(diag * 1e-5, 1e-6)

    touching = []
    for face in faces:
        fb = face.BoundingBox()
        fmin = [fb.xmin, fb.ymin, fb.zmin]
        fmax = [fb.xmax, fb.ymax, fb.zmax]
        touching.append(bool(
            any(abs(fmin[k] - bmin[k]) <= tol for k in range(3))
            or any(abs(fmax[k] - bmax[k]) <= tol for k in range(3))
        ))
    return touching


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------

def _edge_index_lookup(edges):
    """Face-edge -> index in `edges`. Hash map, with a linear-scan fallback."""
    try:
        from OCP.TopTools import TopTools_IndexedMapOfShape

        shape_map = TopTools_IndexedMapOfShape()
        for edge in edges:
            shape_map.Add(edge.wrapped)

        def lookup(face_edge):
            index = shape_map.FindIndex(face_edge.wrapped)
            return index - 1 if index > 0 else None  # OCCT indices are 1-based

        return lookup
    except Exception:
        def lookup(face_edge):
            for i, edge in enumerate(edges):
                if face_edge.wrapped.IsSame(edge.wrapped):
                    return i
            return None

        return lookup


def extract_faces(solid, solid_index, multi_solid):
    faces = solid.Faces()
    face_data = []
    for i, face in enumerate(faces):
        center = face.Center()
        try:
            normal = face.normalAt(center)
        except Exception:
            normal = cq.Vector(0, 0, 0)
        entry = {
            "id": face_label(solid_index, i, multi_solid),
            "type": face.geomType(),
            "area": round(float(face.Area()), DEFAULT_NDIGITS),
            "centroid": _vec_to_list(center),
            "normal": _vec_to_list(normal),
        }
        params = face_analytic_params(face)
        if params:
            entry["geometry_params"] = params
        face_data.append(entry)
    return faces, face_data


def extract_edges(solid, faces, face_data, solid_index, multi_solid):
    edges = solid.Edges()
    edge_face_adjacency = [set() for _ in edges]

    lookup = _edge_index_lookup(edges)
    for face_idx, face in enumerate(faces):
        for face_edge in face.Edges():
            edge_idx = lookup(face_edge)
            if edge_idx is not None:
                edge_face_adjacency[edge_idx].add(face_idx)

    edge_data = []
    for i, edge in enumerate(edges):
        try:
            start, end = edge.startPoint(), edge.endPoint()
            start_l, end_l = _vec_to_list(start), _vec_to_list(end)
        except Exception:
            start_l = end_l = None  # closed curves have no distinct endpoints
        entry = {
            "id": edge_label(solid_index, i, multi_solid),
            "type": edge.geomType(),
            "length": round(float(edge.Length()), DEFAULT_NDIGITS),
            "midpoint": _vec_to_list(edge.Center()),
            "adjacent_faces": sorted(
                face_label(solid_index, j, multi_solid) for j in edge_face_adjacency[i]
            ),
        }
        if start_l is not None:
            entry["start_point"] = start_l
            entry["end_point"] = end_l
        params = edge_analytic_params(edge)
        if params:
            entry["geometry_params"] = params
        edge_data.append(entry)
    return edges, edge_data, edge_face_adjacency


def compute_face_adjacency(face_data, edge_data):
    adjacency = {f["id"]: set() for f in face_data}
    for e in edge_data:
        adj = e["adjacent_faces"]
        for a in adj:
            for b in adj:
                if a != b and a in adjacency:
                    adjacency[a].add(b)
    return {k: sorted(v) for k, v in adjacency.items()}


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def extract_brep(step_path, include_edges=True, include_adjacency=True):
    """Part summary plus one entry per solid, under the renderer's F#/E# labels."""
    solids = load_solids(step_path)
    multi_solid = len(solids) > 1

    solid_entries = []
    bboxes = []

    for solid_index, solid in enumerate(solids):
        bbox = solid_bbox_dict(solid)
        bboxes.append(bbox)

        faces, face_data = extract_faces(solid, solid_index, multi_solid)
        # must precede any tessellation - see compute_bbox_touching_faces
        touching = compute_bbox_touching_faces(faces, bbox)
        for entry, is_touching in zip(face_data, touching):
            entry["touches_bbox"] = is_touching

        edge_data = []
        if include_edges or include_adjacency:
            _edges, edge_data, _adj = extract_edges(
                solid, faces, face_data, solid_index, multi_solid
            )
            if include_adjacency:
                face_adjacency = compute_face_adjacency(face_data, edge_data)
                for entry in face_data:
                    entry["adjacent_faces"] = face_adjacency[entry["id"]]

        try:
            volume = round(float(solid.Volume()), DEFAULT_NDIGITS)
        except Exception:
            volume = None

        solid_entries.append({
            "solid_id": f"S{solid_index}",
            "num_faces": len(faces),
            "num_edges": len(solid.Edges()),
            "num_vertices": len(solid.Vertices()),
            "volume": volume,
            "centroid": _vec_to_list(solid.Center()),
            "bounding_box": bbox,
            "faces": face_data,
            "edges": edge_data if include_edges else [],
        })

    part_bbox = union_bbox(bboxes)
    return {
        "source_file": step_path,
        "num_solids": len(solids),
        "multi_solid": multi_solid,
        "label_convention": (
            "S<i>F<j> / S<i>E<j> (multi-body part)" if multi_solid
            else "F<j> / E<j> (single-body part)"
        ),
        "part_bounding_box": part_bbox,
        "part_size": [round(part_bbox["max"][k] - part_bbox["min"][k], DEFAULT_NDIGITS)
                      for k in range(3)],
        "total_faces": sum(s["num_faces"] for s in solid_entries),
        "total_edges": sum(s["num_edges"] for s in solid_entries),
        "total_volume": round(sum(s["volume"] or 0 for s in solid_entries), DEFAULT_NDIGITS),
        "solids": solid_entries,
    }
