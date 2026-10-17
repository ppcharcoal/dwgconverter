"""High level raster -> vector pipeline."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import skeletonize

from .fitting import ChainFitter, Seg, best_loop_start, fit_line
from .geometry import Graph
from .raster import binarize, clean_graph, load_image, trace_skeleton


@dataclass
class Options:
    mode: str = "auto"            # auto | centerline | outline
    tolerance: float = 1.5        # max deviation from the source pixels (px)
    threshold: int | None = None  # None = automatic (Otsu)
    ink: str = "auto"             # auto | dark | light
    denoise: bool = False
    min_speck: int = 6            # remove ink blobs smaller than this (px area)
    max_line_width: float = 12.0  # auto mode: strokes thicker than this -> outline
    detect_arcs: bool = True
    snap_orthogonal: bool = True
    join_polylines: bool = False  # centerlines as polylines instead of LINE/ARC
    fill_outlines: bool = False   # add solid HATCH inside outlined shapes


@dataclass
class Result:
    width: int
    height: int
    dpi: float | None
    entities: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
def _line_circle_hit(p, d, cx, cy, r, near):
    """Intersection of line (p, d) with circle closest to `near`."""
    f = p - np.array([cx, cy])
    b = float(f @ d)
    c = float(f @ f) - r * r
    disc = b * b - c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    cands = [p + (-b - s) * d, p + (-b + s) * d]
    return min(cands, key=lambda q: float(np.hypot(*(q - near))))


def _joint(s1: Seg, s2: Seg, raw: np.ndarray, tol: float) -> np.ndarray:
    if s1.corner is not None:
        return s1.corner.copy()
    lim = 3 * tol
    if s1.kind == "line" and s2.kind == "line":
        d1, d2 = s1.d, s2.d
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) > 1e-6:
            t = ((s2.p[0] - s1.p[0]) * d2[1] - (s2.p[1] - s1.p[1]) * d2[0]) / den
            q = s1.p + t * d1
            if np.hypot(*(q - raw)) <= lim:
                return q
        return raw
    if s1.kind != s2.kind:
        ln, ar = (s1, s2) if s1.kind == "line" else (s2, s1)
        q = _line_circle_hit(ln.p, ln.d, ar.cx, ar.cy, ar.r, raw)
        if q is not None and np.hypot(*(q - raw)) <= lim:
            return q
        return ln.p + float((raw - ln.p) @ ln.d) * ln.d
    return raw


def _project(seg: Seg, q: np.ndarray) -> np.ndarray:
    if seg.kind == "line":
        return seg.p + float((q - seg.p) @ seg.d) * seg.d
    v = q - np.array([seg.cx, seg.cy])
    n = np.hypot(*v)
    if n == 0:
        return q
    return np.array([seg.cx, seg.cy]) + v / n * seg.r


def _arc_mid(seg: Seg, P: np.ndarray) -> np.ndarray:
    q = P[(seg.i0 + seg.i1) // 2]
    return _project(seg, q)


def _junction_lines(lst):
    out = []
    for s, q in lst:
        w = float(min(max(s.i1 - s.i0, 1), 40))
        seglen = float(s.i1 - s.i0)
        if s.kind == "line":
            d, base = s.d, s.p
        else:
            rv = q - np.array([s.cx, s.cy])
            nr = np.hypot(*rv)
            if nr == 0:
                continue
            base = np.array([s.cx, s.cy]) + rv / nr * s.r
            d = np.array([-rv[1], rv[0]]) / nr
        out.append((base, np.array([-d[1], d[0]]), w, seglen))
    return out


def _solve_junction(p0, lst, tol):
    lines = _junction_lines(lst)
    while len(lines) >= 2:
        # the prior keeps ill-conditioned (near parallel) cases near p0
        lam = 1e-4
        A = np.eye(2) * lam
        b = lam * p0
        for base, n, w, _ in lines:
            N = w * np.outer(n, n)
            A += N
            b += N @ base
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        res = [abs(float((x - base) @ n)) for base, n, w, _ in lines]
        worst = int(np.argmax(res))
        if res[worst] <= 2 * tol:
            return x, sorted((sl for *_, sl in lines), reverse=True)[1]
        if len(lines) == 2:
            return None
        lines.pop(worst)
    return None


def add_chain(G: Graph, P: np.ndarray, segs: list[Seg], va: int, vb: int,
              tol: float, project_a: bool, project_b: bool, cyclic: bool = False):
    if not segs:
        return
    if project_a:
        G.V[va][:] = _project(segs[0], G.V[va])
    if project_b and vb != va:
        G.V[vb][:] = _project(segs[-1], G.V[vb])
    if cyclic and len(segs) > 1:
        G.V[va][:] = _joint(segs[-1], segs[0], P[0], tol)
    verts = [va]
    for k in range(len(segs) - 1):
        q = _joint(segs[k], segs[k + 1], P[segs[k].i1], tol)
        verts.append(G.add_vertex(q))
    verts.append(vb)
    for k, s in enumerate(segs):
        a, b = verts[k], verts[k + 1]
        if s.kind == "line":
            G.add_line(a, b)
        else:
            G.add_arc(a, b, _arc_mid(s, P))


def fit_closed(G: Graph, P: np.ndarray, fitter: ChainFitter, tol: float):
    if fitter.allow_arcs:
        c = fitter.full_circle(P)
        if c is not None:
            from .geometry import Circle
            G.circles.append(Circle(*c))
            return
    s = best_loop_start(P, tol)
    Q = np.vstack([P[s:], P[:s], P[s:s + 1]])
    segs = fitter.segment(Q)
    if len(segs) == 1 and segs[0].kind == "line":
        # degenerate loop (two overlapping strokes) -> split in half
        segs = fitter.segment(Q[: len(Q) // 2 + 1]) + []
        v = G.add_vertex(Q[0])
        w = G.add_vertex(Q[len(Q) // 2])
        add_chain(G, Q[: len(Q) // 2 + 1], segs, v, w, tol, True, True)
        return
    v = G.add_vertex(Q[0])
    add_chain(G, Q, segs, v, v, tol, False, False, cyclic=True)


# --------------------------------------------------------------------------
def _graph_entities(G: Graph, layer: str, as_polylines: bool, closed_only=False):
    ents = []
    for c in G.circles:
        ents.append(("circle", np.array([c.cx, c.cy]), c.r, layer))
    if as_polylines:
        for vs, ks, closed in G.chains():
            pts = [G.V[v].copy() for v in vs]
            mids = [G.prims[k].mid.copy() if G.prims[k].kind == "arc" else None for k in ks]
            if closed:
                pts = pts[:-1]
            if len(ks) == 1 and not closed:
                p = G.prims[ks[0]]
                if p.kind == "line":
                    ents.append(("line", G.V[p.a].copy(), G.V[p.b].copy(), layer))
                else:
                    ents.append(("arc", G.V[p.a].copy(), p.mid.copy(), G.V[p.b].copy(), layer))
                continue
            ents.append(("poly", pts, mids, closed, layer))
    else:
        for p in G.prims:
            if not p.alive:
                continue
            if p.kind == "line":
                ents.append(("line", G.V[p.a].copy(), G.V[p.b].copy(), layer))
            else:
                ents.append(("arc", G.V[p.a].copy(), p.mid.copy(), G.V[p.b].copy(), layer))
    return ents


def vectorize_mask(ink: np.ndarray, opts: Options, progress=None) -> list:
    tol = float(opts.tolerance)
    fitter = ChainFitter(tol=tol, allow_arcs=opts.detect_arcs)
    fitter.ink = ink
    H, W = ink.shape
    ents = []

    def report(msg):
        if progress:
            progress(msg)

    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    fitter.dist = dist

    # ---------------- split solid areas from thin strokes ------------------
    if opts.mode == "centerline":
        solid = np.zeros_like(ink)
    elif opts.mode == "outline":
        solid = ink.copy()
    else:
        report("Separating filled areas from lines…")
        k = int(round(opts.max_line_width)) + 1
        k += (k + 1) % 2
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        solid = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
        if solid.any():
            # opening rounds off corners of filled shapes; give the small
            # leftover slivers back to the solid region they belong to
            rest = (ink & ~solid).astype(np.uint8)
            n_r, lab_r, st_r, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
            near = cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            touch = np.zeros(n_r, bool)
            touch[np.unique(lab_r[near & (rest > 0)])] = True
            small = np.maximum(st_r[:, cv2.CC_STAT_WIDTH], st_r[:, cv2.CC_STAT_HEIGHT]) <= k
            absorb = touch & small
            absorb[0] = False
            solid = solid | absorb[lab_r]
    center_mask = ink & ~solid

    # ---------------- centerlines ----------------------------------------
    if center_mask.any():
        report("Tracing centerlines…")
        sg = trace_skeleton(center_mask, dist)
        clean_graph(sg, tol)
        G = Graph()
        deg = [0] * len(sg.node_pos)
        for e in sg.edges:
            if e["alive"]:
                deg[e["a"]] += 1
                deg[e["b"]] += 1
        vid = {}
        node_ends: dict[int, list] = {}

        def node_v(i):
            if i not in vid:
                vid[i] = G.add_vertex(sg.node_pos[i])
            return vid[i]

        report("Fitting lines and arcs…")
        for e in sg.edges:
            if not e["alive"]:
                continue
            P = e["pts"]
            if len(P) < 2:
                continue
            a, b = e["a"], e["b"]
            if a == b and deg[a] <= 2:
                fit_closed(G, P[:-1], fitter, tol)
                continue
            # skeletons bend near junctions; ignore those pixels while fitting
            # and let the junction position be solved from the clean strokes
            Pt = P
            for kr, kt in ((2.0, 2.0), (1.0, 0.5)):
                i0, i1 = 0, len(P)
                if deg[a] >= 3:
                    far = np.hypot(*(P - sg.node_pos[a]).T) > kr * sg.node_rad[a] + kt * tol
                    i0 = int(np.argmax(far)) if far.any() else 0
                if deg[b] >= 3:
                    far = np.hypot(*(P - sg.node_pos[b]).T) > kr * sg.node_rad[b] + kt * tol
                    i1 = len(P) - int(np.argmax(far[::-1])) if far.any() else len(P)
                if i1 - i0 >= max(8, 0.6 * len(P)):
                    Pt = P[i0:i1]
                    break
            segs = fitter.segment(Pt)
            if not segs:
                continue
            add_chain(G, Pt, segs, node_v(a), node_v(b), tol,
                      project_a=deg[a] == 1, project_b=deg[b] == 1)
            if deg[a] >= 3:
                node_ends.setdefault(a, []).append((segs[0], Pt[0]))
            if deg[b] >= 3:
                node_ends.setdefault(b, []).append((segs[-1], Pt[-1]))
        for P in sg.loops:
            fit_closed(G, P, fitter, tol)

        # solve junction positions: least-squares intersection of the
        # strokes meeting there (outliers such as crossing bridges dropped)
        ink_d = cv2.dilate(center_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for nd, lst in node_ends.items():
            sol = _solve_junction(sg.node_pos[nd], lst, tol)
            if sol is None:
                continue
            x, len2 = sol
            xi, yi = int(round(x[0])), int(round(x[1]))
            if not (0 <= xi < W and 0 <= yi < H and ink_d[yi, xi]):
                continue
            if np.hypot(*(x - sg.node_pos[nd])) > max(60.0, 4 * sg.node_rad[nd]):
                continue
            # the junction may only slide along inked stroke
            if not Graph._ink_between(ink_d, sg.node_pos[nd], x, 0.95):
                continue
            G.V[vid[nd]][:] = x

        report("Merging collinear segments…")
        G.collapse_short(tol)
        G.merge(tol)
        G.merge_collinear_geometric(ink_d, tol)
        G.merge_arcs_geometric(ink_d, tol)
        G.remove_covered_lines(tol)
        G.merge(tol)
        G.merge_collinear_geometric(ink_d, tol)
        if opts.snap_orthogonal:
            G.snap_orthogonal(tol)
        G.project_interior()
        G.remove_degenerate(tol)
        ents += _graph_entities(G, "CENTERLINES", opts.join_polylines)

    # ---------------- outlines -------------------------------------------
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid.astype(np.uint8), connectivity=8)
    if n > 1:
        report("Tracing outlines…")
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        sub = (lab[y:y + h, x:x + w] == c)
        pad = np.zeros((h + 2, w + 2), float)
        pad[1:-1, 1:-1] = sub
        loops = []
        for cont in find_contours(pad, 0.5):
            if len(cont) < 4:
                continue
            P = np.column_stack([cont[:, 1] - 1 + x, cont[:, 0] - 1 + y])
            if np.allclose(P[0], P[-1]):
                P = P[:-1]
            # drop consecutive duplicates
            keep = np.ones(len(P), bool)
            keep[1:] = np.any(np.diff(P, axis=0) != 0, axis=1)
            P = P[keep]
            if len(P) < 3:
                continue
            G = Graph()
            fit_closed(G, P, fitter, tol)
            G.merge(tol)
            if opts.snap_orthogonal:
                G.snap_orthogonal(tol)
            G.project_interior()
            loop_ents = _graph_entities(G, "OUTLINES", True)
            ents += loop_ents
            loops += loop_ents
        if opts.fill_outlines and loops:
            ents.append(("hatch", loops, "FILL"))
    return ents


def vectorize_file(path: str, opts: Options, progress=None) -> Result:
    t0 = time.time()
    gray, dpi = load_image(path)
    if progress:
        progress("Binarizing…")
    ink = binarize(gray, opts.threshold, opts.ink, opts.denoise, opts.min_speck)
    res = vectorize_image_mask(ink, opts, progress)
    res.dpi = dpi
    res.stats["seconds"] = round(time.time() - t0, 2)
    return res


def vectorize_image_mask(ink: np.ndarray, opts: Options, progress=None) -> Result:
    H, W = ink.shape
    ents = vectorize_mask(ink, opts, progress)
    stats = {}
    for e in ents:
        stats[e[0]] = stats.get(e[0], 0) + 1
    return Result(width=W, height=H, dpi=None, entities=ents, stats=stats)
