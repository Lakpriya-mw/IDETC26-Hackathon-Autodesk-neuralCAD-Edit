"""
STEP -> mesh-rendered views with bounding-box and face-label overlays.

All bodies share one image and one camera; labels are solid-prefixed (S0F3)
to match brep_extract. Fallback renderer - vtk_render is used when available.

draw_edges is off by default: hidden-line removal is adjacency-based, and these
parts are mostly curved faces whose centre normal cannot decide visibility, so
the overlay reads as an X-ray. A depth buffer would fix it.
"""

import itertools
import os

import cadquery as cq
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe; no display or GPU required

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from harness.cq.brep_extract import (  # noqa: E402
    compute_bbox_touching_faces,
    face_label,
    load_solids,
    solid_bbox_dict,
    union_bbox,
)

# ---------------------------------------------------------------------------
# camera convention
# ---------------------------------------------------------------------------
ALL_VIEWS = {
    "top":    {"eye_dir": np.array([0.0, 0.0, 1.0]),  "up": np.array([0.0, 1.0, 0.0])},
    "bottom": {"eye_dir": np.array([0.0, 0.0, -1.0]), "up": np.array([0.0, -1.0, 0.0])},
    "front":  {"eye_dir": np.array([0.0, -1.0, 0.0]), "up": np.array([0.0, 0.0, 1.0])},
    "back":   {"eye_dir": np.array([0.0, 1.0, 0.0]),  "up": np.array([0.0, 0.0, 1.0])},
    "left":   {"eye_dir": np.array([-1.0, 0.0, 0.0]), "up": np.array([0.0, 0.0, 1.0])},
    "right":  {"eye_dir": np.array([1.0, 0.0, 0.0]),  "up": np.array([0.0, 0.0, 1.0])},
    # front-biased isometric corners (-Y), the usual CAD convention
    "iso_top_right":    {"eye_dir": np.array([1.0, -1.0, 1.0]),   "up": np.array([0.0, 0.0, 1.0])},
    "iso_top_left":     {"eye_dir": np.array([-1.0, -1.0, 1.0]),  "up": np.array([0.0, 0.0, 1.0])},
    "iso_bottom_right": {"eye_dir": np.array([1.0, -1.0, -1.0]),  "up": np.array([0.0, 0.0, 1.0])},
    "iso_bottom_left":  {"eye_dir": np.array([-1.0, -1.0, -1.0]), "up": np.array([0.0, 0.0, 1.0])},
    # rear-biased octants (+Y), so the back of a part is reachable too
    "iso_back_top_right":    {"eye_dir": np.array([1.0, 1.0, 1.0]),   "up": np.array([0.0, 0.0, 1.0])},
    "iso_back_top_left":     {"eye_dir": np.array([-1.0, 1.0, 1.0]),  "up": np.array([0.0, 0.0, 1.0])},
    "iso_back_bottom_right": {"eye_dir": np.array([1.0, 1.0, -1.0]),  "up": np.array([0.0, 0.0, 1.0])},
    "iso_back_bottom_left":  {"eye_dir": np.array([-1.0, 1.0, -1.0]), "up": np.array([0.0, 0.0, 1.0])},
}

VIEW_NAMES = list(ALL_VIEWS.keys())

# Benchmark output cameras. Filenames are fixed by the ingest contract; the
# camera vectors were recovered by maximising silhouette IoU against the
# dataset's own images (mean 0.965) rather than taken from
# src/utils/cadquery_rendering.VIEW_PROJECTIONS, which disagrees with it:
# top/bottom are X-up, and toprightiso is (1,1,1), not (1,-1,1) (IoU 0.544).
BENCHMARK_VIEWS = {
    "toprightiso": {"eye_dir": np.array([1.0, 1.0, 1.0]),  "up": np.array([0.0, 1.0, 0.0])},
    "front":       {"eye_dir": np.array([0.0, 0.0, 1.0]),  "up": np.array([0.0, 1.0, 0.0])},
    "back":        {"eye_dir": np.array([0.0, 0.0, -1.0]), "up": np.array([0.0, 1.0, 0.0])},
    "left":        {"eye_dir": np.array([-1.0, 0.0, 0.0]), "up": np.array([0.0, 1.0, 0.0])},
    "right":       {"eye_dir": np.array([1.0, 0.0, 0.0]),  "up": np.array([0.0, 1.0, 0.0])},
    "top":         {"eye_dir": np.array([0.0, 1.0, 0.0]),  "up": np.array([-1.0, 0.0, 0.0])},
    "bottom":      {"eye_dir": np.array([0.0, -1.0, 0.0]), "up": np.array([-1.0, 0.0, 0.0])},
}

# Neutral grey, matching the dataset's own renders.
BENCHMARK_FACE_COLOR = np.array([0.72, 0.72, 0.71])
# Finer than the in-loop default; 0.3 leaves visible facets on curved faces.
BENCHMARK_TESSELLATION_TOLERANCE = 0.04

DEFAULT_RESOLUTION_PX = 900
TESSELLATION_TOLERANCE = 0.3
TESSELLATION_ANGULAR_TOLERANCE = 0.3
PADDING_FRAC = 0.12
LABEL_FONT_SIZE = 7
FACE_COLOR_RGB = np.array([0.55, 0.65, 0.80])
LIGHT_DIR_BIAS = 0.35
TOUCHING_FACE_LABEL_COLOR = "deepskyblue"
REQUESTED_FACE_LABEL_COLOR = "orange"   # faces the caller named via label_ids
BBOX_EDGE_COLOR = "black"
BBOX_VERTEX_LABEL_DECIMALS = 3
EDGE_LINE_COLOR = (0.12, 0.12, 0.16)
EDGE_LINE_WIDTH = 0.45
EDGE_SAMPLES_CURVED = 14
MAX_LABELS_PER_VIEW = 60
# Curved-face cull threshold; 0 would erase every hole rim (normal is
# sampled at the face centre, so a cylinder edge-on reads as back-facing).
CURVED_FACE_EDGE_TOLERANCE = -0.35


# ---------------------------------------------------------------------------
# camera / projection
# ---------------------------------------------------------------------------

def _camera_basis(eye_dir, up):
    eye_dir = eye_dir / np.linalg.norm(eye_dir)
    right = np.cross(up, eye_dir)
    right = right / np.linalg.norm(right)
    true_up = np.cross(eye_dir, right)
    return right, true_up, eye_dir


def _project(points, origin, right, true_up, eye_dir):
    rel = points - origin
    return rel @ right, rel @ true_up, rel @ eye_dir


# ---------------------------------------------------------------------------
# bounding-box wireframe (correct for orthographic AND isometric cameras)
# ---------------------------------------------------------------------------

def _bbox_corners_and_edges(bbox):
    axis_vals = [(bbox["min"][k], bbox["max"][k]) for k in range(3)]
    combos = list(itertools.product([0, 1], repeat=3))
    points = {c: np.array([axis_vals[k][c[k]] for k in range(3)]) for c in combos}
    edges = [
        (a, b) for a, b in itertools.combinations(combos, 2)
        if sum(x != y for x, y in zip(a, b)) == 1
    ]
    return points, edges


def draw_bbox_overlay(ax, bbox, origin, right, true_up, eye_dir_n,
                      decimals=BBOX_VERTEX_LABEL_DECIMALS):
    """Bounding-box wireframe; for collapsed corner pairs, labels the nearer."""
    points, edges = _bbox_corners_and_edges(bbox)

    proj = {}
    for combo, pt in points.items():
        x, y, depth = _project(pt[None, :], origin, right, true_up, eye_dir_n)
        proj[combo] = (float(x[0]), float(y[0]), float(depth[0]), pt)

    ax.add_collection(LineCollection(
        [[(proj[a][0], proj[a][1]), (proj[b][0], proj[b][1])] for a, b in edges],
        colors=BBOX_EDGE_COLOR, linewidths=1.1, linestyles="--", zorder=8,
    ))

    groups = {}
    for _combo, (x, y, depth, pt3d) in proj.items():
        key = (round(x, 6), round(y, 6))
        if key not in groups or depth > groups[key][2]:
            groups[key] = (x, y, depth, pt3d)

    for x, y, _depth, pt3d in groups.values():
        ax.plot(x, y, marker="s", markersize=5, color="black", zorder=12)
        ax.annotate(
            f"({pt3d[0]:.{decimals}f}, {pt3d[1]:.{decimals}f}, {pt3d[2]:.{decimals}f})",
            (x, y), textcoords="offset points", xytext=(6, -10),
            fontsize=6, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black",
                      linewidth=0.5, alpha=0.9),
            zorder=13,
        )


# ---------------------------------------------------------------------------
# geometry gathering
# ---------------------------------------------------------------------------

def _tessellate(solid, tolerance=None, angular=None):
    vertices, triangles = solid.tessellate(
        TESSELLATION_TOLERANCE if tolerance is None else tolerance,
        TESSELLATION_ANGULAR_TOLERANCE if angular is None else angular,
    )
    verts = np.array([[v.x, v.y, v.z] for v in vertices])
    tris = np.array(triangles)
    return verts, tris


def _sample_edge(edge):
    """Polyline approximation of an edge: 2 points for a line, more for curves."""
    try:
        if edge.geomType() == "LINE":
            return np.array([edge.startPoint().toTuple(), edge.endPoint().toTuple()])
        n = EDGE_SAMPLES_CURVED
        return np.array([edge.positionAt(i / (n - 1)).toTuple() for i in range(n)])
    except Exception:
        return None


def _safe_normal(face):
    try:
        n = face.normalAt(face.Center())
        return np.array([n.x, n.y, n.z])
    except Exception:
        return None


def _edge_adjacent_normals(solid):
    """Per edge, the adjacent faces' normals and planarity, for edge culling."""
    from harness.cq.brep_extract import _edge_index_lookup

    edges = solid.Edges()
    lookup = _edge_index_lookup(edges)
    adjacency = [[] for _ in edges]

    for face in solid.Faces():
        normal = _safe_normal(face)
        if normal is None:
            continue
        try:
            planar = face.geomType() == "PLANE"
        except Exception:
            planar = False
        for face_edge in face.Edges():
            idx = lookup(face_edge)
            if idx is not None:
                adjacency[idx].append((normal, planar))

    return edges, adjacency


def _gather(step_path, want_edges, want_labels, tolerance=None):
    """Load once; collect the mesh, bbox, label anchors and edge polylines."""
    solids = load_solids(step_path)
    multi = len(solids) > 1

    bboxes = [solid_bbox_dict(s) for s in solids]
    part_bbox = union_bbox(bboxes)

    # Before tessellation: face.BoundingBox() becomes mesh-derived after.
    # All faces, not just bbox-touching - label_ids can name any of them.
    labels = []
    if want_labels:
        for si, solid in enumerate(solids):
            faces = solid.Faces()
            touching = compute_bbox_touching_faces(faces, bboxes[si])
            for fi, (face, is_touching) in enumerate(zip(faces, touching)):
                try:
                    centroid = face.Center()
                    normal = face.normalAt(centroid)
                except Exception:
                    continue
                labels.append({
                    "id": face_label(si, fi, multi),
                    "centroid": [centroid.x, centroid.y, centroid.z],
                    "normal": [normal.x, normal.y, normal.z],
                    "area": float(face.Area()),
                    "touches_bbox": bool(is_touching),
                })

    edge_polys = []
    if want_edges:
        for solid in solids:
            edges, adjacency = _edge_adjacent_normals(solid)
            for edge, adj in zip(edges, adjacency):
                pts = _sample_edge(edge)
                if pts is not None and len(pts) >= 2:
                    edge_polys.append((pts, adj))

    all_verts, all_tris = [], []
    offset = 0
    for solid in solids:
        verts, tris = _tessellate(solid, tolerance=tolerance)
        if len(verts) == 0 or len(tris) == 0:
            continue
        all_verts.append(verts)
        all_tris.append(np.array(tris) + offset)
        offset += len(verts)

    if not all_verts:
        raise ValueError("nothing tessellated - the STEP produced no drawable mesh")

    return {
        "verts": np.vstack(all_verts),
        "tris": np.vstack(all_tris),
        "part_bbox": part_bbox,
        "labels": labels,
        "edge_polys": edge_polys,
        "num_solids": len(solids),
        "multi_solid": multi,
    }


# ---------------------------------------------------------------------------
# shading
# ---------------------------------------------------------------------------

def render_benchmark_views(step_path, output_dir, prefix="tmp", views=None,
                           size=1024, ext="png"):
    """The 7 output views: plain shaded orthographic, framed on the part."""
    global FACE_COLOR_RGB

    os.makedirs(output_dir, exist_ok=True)
    views = [v for v in (views or list(BENCHMARK_VIEWS)) if v in BENCHMARK_VIEWS]
    if not views:
        return []

    geo = _gather(step_path, want_edges=False, want_labels=False,
                  tolerance=BENCHMARK_TESSELLATION_TOLERANCE)

    bbox = geo["part_bbox"]
    origin = np.array([(bbox["min"][k] + bbox["max"][k]) / 2.0 for k in range(3)])
    diag = np.array(bbox["max"]) - np.array(bbox["min"])
    extent = float(np.linalg.norm(diag)) / 2.0 or 1.0

    previous_color = FACE_COLOR_RGB
    FACE_COLOR_RGB = BENCHMARK_FACE_COLOR
    written = []
    try:
        for view_name in views:
            path = os.path.join(output_dir, f"{prefix}_{view_name}.{ext}")
            render = _shade_and_project(BENCHMARK_VIEWS[view_name], geo["verts"],
                                        geo["tris"], origin, extent)

            fig_size = size / 100
            fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
            ax.add_collection(PolyCollection(render["polys"],
                                             facecolors=render["colors"],
                                             edgecolors="none", closed=True))
            ax.set_xlim(-render["half_extent"], render["half_extent"])
            ax.set_ylim(-render["half_extent"], render["half_extent"])
            ax.set_aspect("equal")
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(path, dpi=100, facecolor="white")
            plt.close(fig)

            if os.path.exists(path):
                written.append(path)
    finally:
        FACE_COLOR_RGB = previous_color

    return written


def _shade_and_project(view_cfg, verts, tris, origin, scale_extent):
    right, true_up, eye_dir_n = _camera_basis(view_cfg["eye_dir"], view_cfg["up"])

    tri_verts = verts[tris]
    tri_normals = np.cross(tri_verts[:, 1] - tri_verts[:, 0],
                           tri_verts[:, 2] - tri_verts[:, 0])
    norms = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tri_normals = tri_normals / norms

    facing = tri_normals @ eye_dir_n
    visible = facing > 0                       # backface culling
    vis_tris = tri_verts[visible]
    vis_normals = tri_normals[visible]
    vis_facing = facing[visible]

    sky = np.array([0.35, 0.35, 0.87])
    sky = sky / np.linalg.norm(sky)
    sky_term = np.clip(vis_normals @ sky, 0, 1)
    intensity = np.clip(
        0.25 + (1 - LIGHT_DIR_BIAS) * vis_facing + LIGHT_DIR_BIAS * sky_term, 0.15, 1.0
    )

    flat = vis_tris.reshape(-1, 3)
    fx, fy, fdepth = _project(flat, origin, right, true_up, eye_dir_n)
    fx, fy, fdepth = fx.reshape(-1, 3), fy.reshape(-1, 3), fdepth.reshape(-1, 3)

    order = np.argsort(fdepth.mean(axis=1))    # painter's algorithm, far to near
    polys = np.stack([fx[order], fy[order]], axis=-1)
    colors = np.clip(FACE_COLOR_RGB[None, :] * intensity[order][:, None], 0, 1)

    return {
        "polys": polys,
        "colors": colors,
        "right": right,
        "true_up": true_up,
        "eye_dir_n": eye_dir_n,
        "half_extent": scale_extent * (1 + PADDING_FRAC),
    }


def _draw_edges(ax, edge_polys, origin, right, true_up, eye_dir_n, half_extent):
    """Draw camera-facing edges. Adjacency-based, not a depth buffer."""
    segments = []
    for pts, adjacency in edge_polys:
        if adjacency:
            visible = False
            for normal, planar in adjacency:
                facing = float(normal @ eye_dir_n)
                if facing > (0.0 if planar else CURVED_FACE_EDGE_TOLERANCE):
                    visible = True
                    break
            if not visible:
                continue

        x, y, _depth = _project(pts, origin, right, true_up, eye_dir_n)
        for i in range(len(pts) - 1):
            segments.append([(x[i], y[i]), (x[i + 1], y[i + 1])])

    if segments:
        ax.add_collection(LineCollection(
            segments, colors=[EDGE_LINE_COLOR], linewidths=EDGE_LINE_WIDTH, zorder=6
        ))


def _select_labels(geo, label_faces, label_ids):
    """(chosen, missing_ids, used_explicit). label_ids wins over the envelope."""
    if label_ids:
        wanted = [str(i).strip().upper() for i in label_ids if str(i).strip()]
        by_id = {f["id"].upper(): f for f in geo["labels"]}
        chosen = [by_id[i] for i in wanted if i in by_id]
        missing = [i for i in wanted if i not in by_id]
        return chosen[:MAX_LABELS_PER_VIEW], missing, True

    if not label_faces:
        return [], [], False

    touching = [f for f in geo["labels"] if f.get("touches_bbox")]
    return (sorted(touching, key=lambda f: -f["area"])[:MAX_LABELS_PER_VIEW],
            [], False)


def _render_one_view(view_name, geo, origin, extent, output_path,
                     draw_bbox, label_faces, draw_edges, resolution,
                     label_ids=None):
    render = _shade_and_project(ALL_VIEWS[view_name], geo["verts"], geo["tris"],
                                origin, extent)

    fig_size = resolution / 100
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
    ax.add_collection(PolyCollection(render["polys"], facecolors=render["colors"],
                                     edgecolors="none", closed=True))
    ax.set_xlim(-render["half_extent"], render["half_extent"])
    ax.set_ylim(-render["half_extent"], render["half_extent"])
    ax.set_aspect("equal")
    ax.axis("off")

    if draw_edges and geo["edge_polys"]:
        _draw_edges(ax, geo["edge_polys"], origin, render["right"],
                    render["true_up"], render["eye_dir_n"], render["half_extent"])

    if draw_bbox:
        draw_bbox_overlay(ax, geo["part_bbox"], origin, render["right"],
                          render["true_up"], render["eye_dir_n"])

    chosen, missing, explicit = _select_labels(geo, label_faces, label_ids)
    labelled = len(chosen)

    if chosen:
        # Requested faces get their own colour.
        colour = REQUESTED_FACE_LABEL_COLOR if explicit else TOUCHING_FACE_LABEL_COLOR
        centroids = np.array([f["centroid"] for f in chosen])
        cx, cy, _ = _project(centroids, origin, render["right"],
                             render["true_up"], render["eye_dir_n"])
        for i, face in enumerate(chosen):
            ax.plot(cx[i], cy[i], marker="o", markersize=5 if explicit else 4,
                    color=colour, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=10)
            ax.annotate(
                face["id"], (cx[i], cy[i]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=LABEL_FONT_SIZE, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc=colour,
                          ec="none", alpha=0.85),
                zorder=11,
            )

    title = view_name
    if explicit:
        title += f"  (tagging {labelled} requested face(s))"
    if geo["multi_solid"]:
        title += f"  ({geo['num_solids']} bodies)"
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return labelled, missing


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def render_views(step_path, output_dir, views, draw_bbox=True, label_faces=True,
                 draw_edges=False, resolution=DEFAULT_RESOLUTION_PX, prefix="view",
                 label_ids=None):
    """Render `views` of `step_path`. label_ids tags those faces specifically."""
    os.makedirs(output_dir, exist_ok=True)

    unknown = [v for v in views if v not in ALL_VIEWS]
    if unknown:
        raise ValueError(
            f"unknown view(s) {unknown}. Valid views: {', '.join(VIEW_NAMES)}"
        )

    want_labels = bool(label_faces or label_ids)
    geo = _gather(step_path, want_edges=draw_edges, want_labels=want_labels)

    bbox = geo["part_bbox"]
    origin = np.array([(bbox["min"][k] + bbox["max"][k]) / 2.0 for k in range(3)])
    diag = np.array(bbox["max"]) - np.array(bbox["min"])
    extent = float(np.linalg.norm(diag)) / 2.0 or 1.0

    images, labelled, missing = [], 0, []
    for view_name in views:
        path = os.path.join(output_dir, f"{prefix}_{view_name}.png")
        labelled, missing = _render_one_view(
            view_name, geo, origin, extent, path,
            draw_bbox, label_faces, draw_edges, resolution, label_ids,
        )
        if os.path.exists(path):
            images.append(path)

    return {
        "images": images,
        "num_solids": geo["num_solids"],
        "multi_solid": geo["multi_solid"],
        "labelled_faces": labelled,
        "missing_ids": missing,
        "part_bounding_box": bbox,
    }
