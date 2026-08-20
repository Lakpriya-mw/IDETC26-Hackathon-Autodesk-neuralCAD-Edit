"""
Rendering through OpenCascade's own VTK pipeline.

Used for every image the pipeline produces - opening views, in-loop build
previews, and the 7 deliverables - so quality does not depend on where in the
pipeline you are. The kernel's tessellation is fine and smooth-shaded, unlike
brep_render's coarse flat-shaded mesh (kept as fallback, and owns the camera
table).

VTK cannot draw the bounding box or face ids, so those are composited after the
render, projected through the camera VTK actually used (see _projector).

cadquery.vis.show is not used: it applies roll=-35/elevation=-45 after
positioning the camera, and exposes no parallel-scale control.
"""

import itertools
import os

import cadquery as cq
import numpy as np
from cadquery.vis import toVTKAssy
from PIL import Image, ImageDraw, ImageFont
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkWindowToImageFilter,
)

from harness.cq.brep_extract import (
    compute_bbox_touching_faces,
    face_label,
    load_solids,
    solid_bbox_dict,
    union_bbox,
)
from harness.cq.brep_render import ALL_VIEWS, BENCHMARK_VIEWS

# The dataset's grey; the kernel default is yellow.
FACE_RGBA = (0.75, 0.75, 0.74, 1.0)
EDGE_RGBA = (0.05, 0.05, 0.05, 1.0)
EDGE_WIDTH = 1.4

# Deflection is absolute, so scale it to the part size.
RELATIVE_TOLERANCE = 2.0e-4
ANGULAR_TOLERANCE = 0.06


MAX_LABELS_PER_VIEW = 60
TOUCHING_LABEL_RGB = (0, 191, 255)      # deepskyblue, as in the mesh renderer
REQUESTED_LABEL_RGB = (255, 165, 0)     # orange, for caller-named faces


def _bounds(shape):
    bb = shape.BoundingBox()
    centre = np.array([(bb.xmin + bb.xmax) / 2.0,
                       (bb.ymin + bb.ymax) / 2.0,
                       (bb.zmin + bb.zmax) / 2.0])
    return centre, max(bb.DiagonalLength, 1e-6)


# --- overlays, composited after the render ---------------------------------

def _projector(camera, size):
    """World point -> (x_px, y_px, depth) under VTK's parallel projection."""
    focus = np.array(camera.GetFocalPoint(), dtype=float)
    position = np.array(camera.GetPosition(), dtype=float)
    up = np.array(camera.GetViewUp(), dtype=float)
    scale = float(camera.GetParallelScale())  # half the view HEIGHT, in world units

    eye = position - focus
    eye /= np.linalg.norm(eye)
    right = np.cross(up, eye)
    right /= np.linalg.norm(right)
    true_up = np.cross(eye, right)

    half = size / 2.0

    def project(points):
        rel = np.atleast_2d(np.asarray(points, dtype=float)) - focus
        x = (rel @ right) / scale
        y = (rel @ true_up) / scale
        return half * (1.0 + x), half * (1.0 - y), rel @ eye

    return project


def _font(px):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed_line(draw, p0, p1, fill, width=1, dash=7, gap=5):
    (x0, y0), (x1, y1) = p0, p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-6:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line([(x0 + ux * pos, y0 + uy * pos),
                   (x0 + ux * end, y0 + uy * end)], fill=fill, width=width)
        pos = end + gap


def _place_text(draw, x, y, text, font, size, dx=5, dy=-14, pad=2):
    """Nudge a label inside the frame; the fitted bbox reaches the edge."""
    box = draw.textbbox((x + dx, y + dy), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    tx, ty = x + dx, y + dy
    if tx + w + pad > size:
        tx = x - dx - w          # flip to the other side of the point
    if ty + h + pad > size:
        ty = y - h - abs(dy)
    tx = max(pad, min(tx, size - w - pad))
    ty = max(pad, min(ty, size - h - pad))
    return tx, ty, w, h


def _label_chip(draw, x, y, text, rgb, font, size):
    """A coloured dot plus a boxed id, matching the mesh renderer's styling."""
    r = 3.5
    draw.ellipse([x - r, y - r, x + r, y + r], fill=rgb, outline=(0, 0, 0))
    tx, ty, w, h = _place_text(draw, x, y, text, font, size)
    draw.rectangle([tx - 2, ty - 2, tx + w + 2, ty + h + 2], fill=rgb)
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)


def _draw_bbox(draw, bbox, project, size):
    """Projected bounding-box wireframe with each visible corner's coordinate."""
    axis_vals = [(bbox["min"][k], bbox["max"][k]) for k in range(3)]
    combos = list(itertools.product([0, 1], repeat=3))
    corners = {c: np.array([axis_vals[k][c[k]] for k in range(3)]) for c in combos}

    proj = {}
    for combo, pt in corners.items():
        px, py, dz = project(pt[None, :])
        proj[combo] = (float(px[0]), float(py[0]), float(dz[0]), pt)

    for a, b in itertools.combinations(combos, 2):
        if sum(i != j for i, j in zip(a, b)) == 1:      # box edges, not diagonals
            _dashed_line(draw, proj[a][:2], proj[b][:2], (30, 30, 30), width=1)

    # An axis-aligned view collapses corner pairs onto one point; label the nearer.
    groups = {}
    for _combo, (px, py, dz, pt) in proj.items():
        key = (round(px, 1), round(py, 1))
        if key not in groups or dz > groups[key][2]:
            groups[key] = (px, py, dz, pt)

    font = _font(11)
    for px, py, _dz, pt in groups.values():
        draw.rectangle([px - 3, py - 3, px + 3, py + 3], fill=(0, 0, 0))
        text = f"({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})"
        tx, ty, w, h = _place_text(draw, px, py, text, font, size, dx=6, dy=4)
        draw.rectangle([tx - 2, ty - 2, tx + w + 2, ty + h + 2],
                       fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)


def _label_anchors(step_path, want_labels):
    """Face ids and centroids. No tessellation - the bbox test needs exact
    surfaces, and VTK meshes separately."""
    if not want_labels:
        return [], None

    solids = load_solids(step_path)
    multi = len(solids) > 1
    boxes = [solid_bbox_dict(s) for s in solids]

    anchors = []
    for si, solid in enumerate(solids):
        faces = solid.Faces()
        touching = compute_bbox_touching_faces(faces, boxes[si])
        for fi, (face, is_touching) in enumerate(zip(faces, touching)):
            try:
                centre = face.Center()
                area = float(face.Area())
            except Exception:
                continue
            anchors.append({
                "id": face_label(si, fi, multi),
                "centroid": [centre.x, centre.y, centre.z],
                "area": area,
                "touches_bbox": bool(is_touching),
            })

    return anchors, union_bbox(boxes)


def _build_scene(step_path, diag, draw_edges):
    """VTK actors, meshed at a part-relative deflection."""
    shape = cq.importers.importStep(step_path)
    return toVTKAssy(
        cq.Assembly(shape),
        color=FACE_RGBA,
        edgecolor=EDGE_RGBA,
        edges=bool(draw_edges),
        linewidth=EDGE_WIDTH,
        tolerance=max(diag * RELATIVE_TOLERANCE, 1e-5),
        angularTolerance=ANGULAR_TOLERANCE,
    )


def render_annotated_views(step_path, output_dir, prefix="view", views=None,
                           size=900, ext="png", draw_bbox=True, label_faces=True,
                           label_ids=None, draw_edges=True, view_table=None):
    """
    Kernel-quality views with bbox and face labels composited on top - what the
    agent sees, in the opening message and from `step_to_views`.

    `view_table`: ALL_VIEWS (default, 14) or BENCHMARK_VIEWS (the 7 outputs).
    """
    os.makedirs(output_dir, exist_ok=True)
    table = view_table or ALL_VIEWS
    views = [v for v in (views or ["iso_top_right"]) if v in table]
    if not views:
        return {"images": [], "missing_ids": [], "labelled_faces": 0,
                "num_solids": 0, "multi_solid": False}

    solids = load_solids(step_path)
    multi = len(solids) > 1
    part_bbox = union_bbox([solid_bbox_dict(s) for s in solids])
    centre = np.array([(part_bbox["min"][k] + part_bbox["max"][k]) / 2.0
                       for k in range(3)])
    diag = float(np.linalg.norm(np.array(part_bbox["max"]) - np.array(part_bbox["min"])))
    diag = max(diag, 1e-6)

    want_labels = bool(label_faces or label_ids)
    anchors, _ = _label_anchors(step_path, want_labels)

    missing = []
    if label_ids:
        wanted = [str(i).strip().upper() for i in label_ids if str(i).strip()]
        by_id = {a["id"].upper(): a for a in anchors}
        chosen = [by_id[i] for i in wanted if i in by_id]
        missing = [i for i in wanted if i not in by_id]
        label_rgb = REQUESTED_LABEL_RGB
    elif label_faces:
        chosen = sorted([a for a in anchors if a["touches_bbox"]],
                        key=lambda a: -a["area"])
        label_rgb = TOUCHING_LABEL_RGB
    else:
        chosen, label_rgb = [], TOUCHING_LABEL_RGB
    chosen = chosen[:MAX_LABELS_PER_VIEW]

    renderer = vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    for prop in _build_scene(step_path, diag, draw_edges):
        renderer.AddActor(prop)

    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(int(size), int(size))
    window.SetMultiSamples(8)

    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    font = _font(12)
    written = []
    for view in views:
        cfg = table[view]
        direction = np.asarray(cfg["eye_dir"], dtype=float)
        direction /= np.linalg.norm(direction)

        camera.SetFocalPoint(*centre)
        camera.SetPosition(*(centre + direction * diag))
        camera.SetViewUp(*np.asarray(cfg["up"], dtype=float))
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()
        window.Render()

        grabber = vtkWindowToImageFilter()
        grabber.SetInput(window)
        grabber.ReadFrontBufferOff()
        grabber.Update()

        path = os.path.join(output_dir, f"{prefix}_{view}.{ext}")
        writer = vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(grabber.GetOutputPort())
        writer.Write()

        if not os.path.exists(path):
            continue

        if draw_bbox or chosen:
            project = _projector(camera, size)
            image = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(image)

            if draw_bbox:
                _draw_bbox(draw, part_bbox, project, size)

            if chosen:
                pts = np.array([a["centroid"] for a in chosen])
                xs, ys, _ = project(pts)
                for anchor, x, y in zip(chosen, xs, ys):
                    _label_chip(draw, float(x), float(y), anchor["id"],
                                label_rgb, font, size)

            title = view + (f"  ({len(solids)} bodies)" if multi else "")
            draw.text((10, 8), title, fill=(40, 40, 40), font=_font(14))
            image.save(path)

        written.append(path)

    window.Finalize()
    return {
        "images": written,
        "missing_ids": missing,
        "labelled_faces": len(chosen),
        "num_solids": len(solids),
        "multi_solid": multi,
        "part_bounding_box": part_bbox,
    }


def render_benchmark_views(step_path, output_dir, prefix="tmp", views=None,
                           size=1024, ext="png", draw_edges=True):
    """The benchmark's views, kernel-tessellated and smooth-shaded."""
    os.makedirs(output_dir, exist_ok=True)
    views = [v for v in (views or list(BENCHMARK_VIEWS)) if v in BENCHMARK_VIEWS]
    if not views:
        return []

    shape = cq.importers.importStep(step_path)
    solid = shape.val() if hasattr(shape, "val") else shape
    centre, diag = _bounds(solid)

    props = toVTKAssy(
        cq.Assembly(shape),
        color=FACE_RGBA,
        edgecolor=EDGE_RGBA,
        edges=bool(draw_edges),
        linewidth=EDGE_WIDTH,
        tolerance=max(diag * RELATIVE_TOLERANCE, 1e-5),
        angularTolerance=ANGULAR_TOLERANCE,
    )

    renderer = vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    for prop in props:
        renderer.AddActor(prop)

    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(int(size), int(size))
    window.SetMultiSamples(8)          # anti-alias the silhouette

    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()      # the dataset's views are parallel

    written = []
    for view in views:
        cfg = BENCHMARK_VIEWS[view]
        direction = np.asarray(cfg["eye_dir"], dtype=float)
        direction = direction / np.linalg.norm(direction)
        up = np.asarray(cfg["up"], dtype=float)

        camera.SetFocalPoint(*centre)
        camera.SetPosition(*(centre + direction * diag))
        camera.SetViewUp(*up)
        # Keeps the direction and up vector, refits distance and parallel
        # scale to the model - the equivalent of the OCC path's FitAll().
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()

        window.Render()

        grabber = vtkWindowToImageFilter()
        grabber.SetInput(window)
        grabber.ReadFrontBufferOff()
        grabber.Update()

        path = os.path.join(output_dir, f"{prefix}_{view}.{ext}")
        writer = vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(grabber.GetOutputPort())
        writer.Write()

        if os.path.exists(path):
            written.append(path)

    window.Finalize()
    return written
