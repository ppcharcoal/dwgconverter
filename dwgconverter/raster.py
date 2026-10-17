"""Image loading, binarisation and skeleton-graph tracing."""
from __future__ import annotations

import numpy as np
import cv2
from skimage.morphology import skeletonize


def load_image(path: str):
    """Load an image as (gray uint8, dpi or None). Handles unicode paths
    and transparency (transparent pixels are treated as background)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, int(img.max())))
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = bgr * a + 255.0 * (1 - a)
        gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dpi = None
    try:
        from PIL import Image
        with Image.open(path) as im:
            d = im.info.get("dpi")
            if d and d[0] and float(d[0]) > 1:
                dpi = float(d[0])
    except Exception:
        pass
    return gray, dpi


def binarize(gray: np.ndarray, threshold: int | None = None,
             invert: str = "auto", denoise: bool = False,
             min_speck: int = 4) -> np.ndarray:
    """Return a bool mask where True = ink.

    threshold None -> Otsu.  invert: 'auto' (ink = minority colour),
    'dark' (dark ink on light paper) or 'light' (light ink on dark)."""
    g = gray
    if denoise:
        # edge-preserving smoothing (keeps 1px lines, removes JPEG mush)
        g = cv2.bilateralFilter(g, 5, 40, 5)
    if threshold is None:
        t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = threshold
    dark = g <= t
    if invert == "dark":
        ink = dark
    elif invert == "light":
        ink = ~dark
    else:
        ink = dark if dark.mean() <= 0.5 else ~dark
    if min_speck > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < min_speck
        small[0] = False
        ink = ink & ~small[lab]
    return ink


# --------------------------------------------------------------------------
# Skeleton graph
# --------------------------------------------------------------------------
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class SkelGraph:
    """Nodes (endpoints/junction clusters) and edges (pixel paths)."""

    def __init__(self):
        self.node_pos: list[np.ndarray] = []   # (x, y) image coords
        self.node_rad: list[float] = []        # local stroke half-width
        self.edges: list[dict] = []            # {a, b, pts (Nx2 x,y), alive}
        self.loops: list[np.ndarray] = []      # closed paths without nodes


def trace_skeleton(mask: np.ndarray, dist: np.ndarray) -> SkelGraph:
    sk = skeletonize(mask).astype(np.uint8)
    H, W = sk.shape
    skp = np.zeros((H + 2, W + 2), np.uint8)
    skp[1:-1, 1:-1] = sk
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    deg = cv2.filter2D(skp, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT)
    deg = deg * skp
    node_mask = ((deg != 2) & (skp > 0)).astype(np.uint8)
    nlab_n, nlab = cv2.connectedComponents(node_mask, connectivity=8)

    g = SkelGraph()
    distp = np.zeros((H + 2, W + 2), np.float32)
    distp[1:-1, 1:-1] = dist
    # node centroids
    ys, xs = np.nonzero(node_mask)
    labs = nlab[ys, xs]
    cnt = np.bincount(labs, minlength=nlab_n).astype(float)
    sx = np.bincount(labs, weights=xs, minlength=nlab_n)
    sy = np.bincount(labs, weights=ys, minlength=nlab_n)
    sr = np.zeros(nlab_n)
    np.maximum.at(sr, labs, distp[ys, xs])
    node_id = {}
    for lab in range(1, nlab_n):
        if cnt[lab] == 0:
            continue
        node_id[lab] = len(g.node_pos)
        # -1 removes the padding; image coords
        g.node_pos.append(np.array([sx[lab] / cnt[lab] - 1, sy[lab] / cnt[lab] - 1]))
        g.node_rad.append(float(sr[lab]))

    visited = np.zeros_like(skp, dtype=bool)

    def neighbours(y, x):
        for dy, dx in _NB:
            if skp[y + dy, x + dx]:
                yield y + dy, x + dx

    # edges starting at nodes
    for y0, x0 in zip(ys, xs):
        a = node_id[nlab[y0, x0]]
        for y1, x1 in neighbours(y0, x0):
            if node_mask[y1, x1]:
                continue
            if visited[y1, x1]:
                continue
            path = [(x1 - 1, y1 - 1)]
            visited[y1, x1] = True
            py, px = y0, x0
            cy, cx = y1, x1
            end = None
            while True:
                nxt = None
                for ny, nx in neighbours(cy, cx):
                    if (ny, nx) == (py, px):
                        continue
                    if node_mask[ny, nx]:
                        # prefer finishing at a node only if it's not where we came from
                        if nlab[ny, nx] != nlab[py, px] or len(path) > 1:
                            end = (ny, nx)
                        continue
                    if not visited[ny, nx]:
                        nxt = (ny, nx)
                if nxt is None:
                    break
                py, px = cy, cx
                cy, cx = nxt
                visited[cy, cx] = True
                path.append((cx - 1, cy - 1))
            if end is None:
                # dead end against already visited pixels -> close at start node
                end = (y0, x0)
            b = node_id[nlab[end[0], end[1]]]
            pts = np.vstack([g.node_pos[a], np.array(path, float), g.node_pos[b]])
            g.edges.append(dict(a=a, b=b, pts=pts, alive=True))

    # isolated adjacent node pairs that touch two different node clusters are
    # impossible (8-connected clusters), so remaining pixels are pure loops
    rem = np.argwhere((skp > 0) & ~visited & (node_mask == 0))
    for y0, x0 in rem:
        if visited[y0, x0]:
            continue
        path = [(x0 - 1, y0 - 1)]
        visited[y0, x0] = True
        py, px = -10, -10
        cy, cx = y0, x0
        while True:
            nxt = None
            for ny, nx in neighbours(cy, cx):
                if (ny, nx) != (py, px) and not visited[ny, nx]:
                    nxt = (ny, nx)
                    break
            if nxt is None:
                break
            py, px = cy, cx
            cy, cx = nxt
            visited[cy, cx] = True
            path.append((cx - 1, cy - 1))
        if len(path) >= 4:
            g.loops.append(np.array(path, float))
    return g


def clean_graph(g: SkelGraph, tol: float):
    """Remove skeleton spurs, contract tiny bridges between junctions and
    concatenate edges through degree-2 nodes."""
    def degree():
        d = [0] * len(g.node_pos)
        for e in g.edges:
            if e["alive"]:
                d[e["a"]] += 1
                d[e["b"]] += 1
        return d

    def length(e):
        p = e["pts"]
        return float(np.hypot(*np.diff(p, axis=0).T).sum())

    for _ in range(3):
        changed = False
        deg = degree()
        # spurs and tiny self loops
        for e in g.edges:
            if not e["alive"]:
                continue
            a, b = e["a"], e["b"]
            L = length(e)
            if a == b and L <= max(4 * tol, 2.5 * g.node_rad[a] + 2):
                e["alive"] = False
                changed = True
                continue
            if a != b and (deg[a] == 1) != (deg[b] == 1):
                j = b if deg[a] == 1 else a
                if deg[j] >= 3 and L <= max(2 * tol, 1.5 * g.node_rad[j] + 1):
                    e["alive"] = False
                    changed = True
        # tiny bridges between junctions -> contract
        deg = degree()
        parent = list(range(len(g.node_pos)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in g.edges:
            if not e["alive"] or e["a"] == e["b"]:
                continue
            a, b = e["a"], e["b"]
            if deg[a] >= 3 and deg[b] >= 3:
                L = length(e)
                if L <= max(2 * tol, g.node_rad[a] + g.node_rad[b] + 1):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                        g.node_pos[ra] = (g.node_pos[ra] + g.node_pos[rb]) / 2
                        g.node_rad[ra] = max(g.node_rad[ra], g.node_rad[rb])
                    e["alive"] = False
                    changed = True
        for e in g.edges:
            if e["alive"]:
                e["a"], e["b"] = find(e["a"]), find(e["b"])
                e["pts"][0] = g.node_pos[e["a"]]
                e["pts"][-1] = g.node_pos[e["b"]]
        # join through degree-2 nodes
        deg = degree()
        inc: dict[int, list[dict]] = {}
        for e in g.edges:
            if e["alive"]:
                inc.setdefault(e["a"], []).append(e)
                if e["b"] != e["a"]:
                    inc.setdefault(e["b"], []).append(e)
        for v, es in inc.items():
            es = [e for e in es if e["alive"]]
            if deg[v] != 2 or len(es) != 2:
                continue
            e1, e2 = es
            if e1 is e2:
                continue
            p1 = e1["pts"] if e1["b"] == v else e1["pts"][::-1]
            p2 = e2["pts"] if e2["a"] == v else e2["pts"][::-1]
            a = e1["a"] if e1["b"] == v else e1["b"]
            b = e2["b"] if e2["a"] == v else e2["a"]
            e1["alive"] = e2["alive"] = False
            ne = dict(a=a, b=b, pts=np.vstack([p1, p2[1:]]), alive=True)
            g.edges.append(ne)
            inc.setdefault(a, []).append(ne)
            inc.setdefault(b, []).append(ne)
            if a == b == v:
                pass
            changed = True
        if not changed:
            break
    # edges that became closed loops through a now-unused degree-2 node
    return g
