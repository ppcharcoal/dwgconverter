"""Vector geometry graph: shared vertices + line/arc/circle primitives.

Primitives reference vertex ids, so connected strokes stay exactly
connected while we merge collinear pieces, snap near-orthogonal lines
and re-project T-junctions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prim:
    kind: str                     # "line" | "arc"
    a: int                        # start vertex id
    b: int                        # end vertex id
    mid: np.ndarray | None = None  # arc: a point on the arc between a and b
    alive: bool = True
    interior: list = field(default_factory=list)  # vertex ids lying on a line


@dataclass
class Circle:
    cx: float
    cy: float
    r: float


def circle_from_3(p1, p2, p3):
    """Circumcircle of three points -> (cx, cy, r) or None if collinear."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def arc_params(pa, pm, pb):
    """Arc through pa -> pm -> pb.

    Returns dict(cx, cy, r, ccw, sweep) where sweep is the signed sweep from
    pa to pb (radians, CCW positive), or None if the points are collinear.
    """
    c = circle_from_3(pa, pm, pb)
    if c is None:
        return None
    cx, cy, r = c
    cross = (pm[0] - pa[0]) * (pb[1] - pm[1]) - (pm[1] - pa[1]) * (pb[0] - pm[0])
    ccw = cross > 0
    a0 = math.atan2(pa[1] - cy, pa[0] - cx)
    a1 = math.atan2(pb[1] - cy, pb[0] - cx)
    if ccw:
        sweep = (a1 - a0) % (2 * math.pi)
    else:
        sweep = -((a0 - a1) % (2 * math.pi))
    return dict(cx=cx, cy=cy, r=r, ccw=ccw, sweep=sweep)


class Graph:
    def __init__(self):
        self.V: list[np.ndarray] = []
        self.prims: list[Prim] = []
        self.circles: list[Circle] = []

    # -- construction ------------------------------------------------------
    def add_vertex(self, p) -> int:
        self.V.append(np.asarray(p, dtype=float).copy())
        return len(self.V) - 1

    def add_line(self, a, b):
        if a == b:
            return
        self.prims.append(Prim("line", a, b))

    def add_arc(self, a, b, mid):
        self.prims.append(Prim("arc", a, b, mid=np.asarray(mid, float)))

    # -- helpers -------------------------------------------------------------
    def incidence(self):
        inc: dict[int, list[int]] = {}
        for k, p in enumerate(self.prims):
            if not p.alive:
                continue
            inc.setdefault(p.a, []).append(k)
            inc.setdefault(p.b, []).append(k)
        return inc

    def _dir_from(self, k, v):
        """Unit tangent of prim k leaving vertex v."""
        p = self.prims[k]
        if p.kind == "line":
            o = p.b if p.a == v else p.a
            d = self.V[o] - self.V[v]
        else:
            ap = arc_params(self.V[p.a], p.mid, self.V[p.b])
            if ap is None:
                d = p.mid - self.V[v]
            else:
                rv = self.V[v] - np.array([ap["cx"], ap["cy"]])
                t = np.array([-rv[1], rv[0]])      # CCW tangent
                # direction of travel along the arc leaving v
                forward = ap["ccw"] if v == p.a else not ap["ccw"]
                d = t if forward else -t
        n = np.hypot(*d)
        return d / n if n > 0 else d

    def remove_degenerate(self, tol):
        for p in self.prims:
            if not p.alive:
                continue
            if p.a == p.b and p.kind == "line":
                p.alive = False
                continue
            if np.hypot(*(self.V[p.a] - self.V[p.b])) < 1e-9 and p.kind == "line":
                p.alive = False

    def collapse_short(self, tol):
        """Contract lines shorter than tol into a single vertex."""
        remap = {}

        def rep(x):
            while x in remap:
                x = remap[x]
            return x
        for p in self.prims:
            if not p.alive:
                continue
            p.a, p.b = rep(p.a), rep(p.b)
            if p.kind == "line" and np.hypot(*(self.V[p.a] - self.V[p.b])) < tol:
                a, b = p.a, p.b
                p.alive = False
                if a != b:
                    self.V[a][:] = (self.V[a] + self.V[b]) / 2
                    remap[b] = a
        for p in self.prims:
            if not p.alive:
                continue
            p.a, p.b = rep(p.a), rep(p.b)
            p.interior = [rep(q) for q in p.interior]
            if p.a == p.b:
                p.alive = False

    # -- collinear / co-circular merging -----------------------------------
    def merge(self, tol: float, angle_tol_deg: float = 3.0):
        cos_tol = math.cos(math.radians(angle_tol_deg))
        inc = self.incidence()
        work = list(inc.keys())
        queued = set(work)
        while work:
            v = work.pop()
            queued.discard(v)
            ks = [k for k in inc.get(v, []) if self.prims[k].alive]
            inc[v] = ks
            if len(ks) < 2:
                continue
            merged = False
            for x in range(len(ks)):
                for y in range(x + 1, len(ks)):
                    k1, k2 = ks[x], ks[y]
                    p1, p2 = self.prims[k1], self.prims[k2]
                    if p1.kind != p2.kind:
                        continue
                    n_before = len(self.prims)
                    if p1.kind == "line":
                        ok = self._try_merge_lines(v, k1, k2, tol, cos_tol)
                    else:
                        ok = self._try_merge_arcs(v, k1, k2, tol)
                    if ok:
                        ends = {p1.a, p1.b, p2.a, p2.b}
                        if len(self.prims) > n_before:   # new prim created
                            nk = len(self.prims) - 1
                            np_ = self.prims[nk]
                            inc.setdefault(np_.a, []).append(nk)
                            inc.setdefault(np_.b, []).append(nk)
                        for e in ends:
                            if e not in queued:
                                work.append(e)
                                queued.add(e)
                        merged = True
                        break
                if merged:
                    break

    def _try_merge_lines(self, v, k1, k2, tol, cos_tol):
        p1, p2 = self.prims[k1], self.prims[k2]
        o1 = p1.b if p1.a == v else p1.a
        o2 = p2.b if p2.a == v else p2.a
        if o1 == o2 or o1 == v or o2 == v:
            return False
        A, B = self.V[o1], self.V[o2]
        # v must lie between the two far ends (lines continue through v)
        if float((A - self.V[v]) @ (B - self.V[v])) >= 0:
            return False
        AB = B - A
        L = np.hypot(*AB)
        if L == 0:
            return False
        n = np.array([-AB[1], AB[0]]) / L
        pts = [v] + p1.interior + p2.interior
        if any(abs(float((self.V[q] - A) @ n)) > tol for q in pts):
            return False
        p1.alive = p2.alive = False
        new = Prim("line", o1, o2, interior=pts)
        self.prims.append(new)
        return True

    def _try_merge_arcs(self, v, k1, k2, tol):
        p1, p2 = self.prims[k1], self.prims[k2]
        o1 = p1.b if p1.a == v else p1.a
        o2 = p2.b if p2.a == v else p2.a
        c1 = arc_params(self.V[p1.a], p1.mid, self.V[p1.b])
        c2 = arc_params(self.V[p2.a], p2.mid, self.V[p2.b])
        if c1 is None or c2 is None:
            return False
        if (math.hypot(c1["cx"] - c2["cx"], c1["cy"] - c2["cy"]) > 2 * tol
                or abs(c1["r"] - c2["r"]) > tol):
            return False
        d1 = self._dir_from(k1, v)
        d2 = self._dir_from(k2, v)
        if float(d1 @ d2) > 0:              # must continue through v
            return False
        sweep = abs(c1["sweep"]) + abs(c2["sweep"])
        if o1 == o2 or sweep > 2 * math.pi - 0.05:
            # closes into a full circle
            p1.alive = p2.alive = False
            self.circles.append(Circle((c1["cx"] + c2["cx"]) / 2,
                                       (c1["cy"] + c2["cy"]) / 2,
                                       (c1["r"] + c2["r"]) / 2))
            return True
        # every original point must still be on the merged arc
        merged = arc_params(self.V[o1], self.V[v], self.V[o2])
        if merged is None:
            return False
        for q in (p1.mid, p2.mid):
            if abs(math.hypot(q[0] - merged["cx"], q[1] - merged["cy"]) - merged["r"]) > tol:
                return False
        p1.alive = p2.alive = False
        self.prims.append(Prim("arc", o1, o2, mid=self.V[v].copy()))
        return True

    # -- geometric (non-topological) merging ----------------------------------
    @staticmethod
    def _ink_between(ink, p, q, min_ratio=0.85):
        """True if the straight path p->q runs over ink (continuous stroke)."""
        if ink is None:
            return False
        L = float(np.hypot(*(q - p)))
        if L < 1.5:
            return True
        n = int(L) + 2
        t = np.linspace(0, 1, n)
        xs = np.clip(np.rint(p[0] + (q[0] - p[0]) * t).astype(int), 0, ink.shape[1] - 1)
        ys = np.clip(np.rint(p[1] + (q[1] - p[1]) * t).astype(int), 0, ink.shape[0] - 1)
        return float(ink[ys, xs].mean()) >= min_ratio

    @staticmethod
    def _ink_along_arc(ink, cx, cy, r, a0, sweep, min_ratio=0.85):
        if ink is None:
            return False
        L = abs(sweep) * r
        if L < 1.5:
            return True
        t = np.linspace(a0, a0 + sweep, int(L) + 2)
        xs = np.clip(np.rint(cx + r * np.cos(t)).astype(int), 0, ink.shape[1] - 1)
        ys = np.clip(np.rint(cy + r * np.sin(t)).astype(int), 0, ink.shape[0] - 1)
        return float(ink[ys, xs].mean()) >= min_ratio

    def merge_collinear_geometric(self, ink, tol, search=60.0, angle_tol_deg=4.0):
        """Join line pieces that lie on one straight stroke even when the
        skeleton topology broke them apart (shallow crossings, merged
        strokes).  Gaps are only bridged when they are covered with ink, so
        dashed lines stay dashed."""
        cos_tol = math.cos(math.radians(angle_tol_deg))
        for _ in range(6):
            rows = [k for k, p in enumerate(self.prims) if p.alive and p.kind == "line"]
            if len(rows) < 2:
                return
            A = np.array([self.V[self.prims[k].a] for k in rows])
            B = np.array([self.V[self.prims[k].b] for k in rows])
            D = B - A
            L = np.hypot(D[:, 0], D[:, 1])
            L[L == 0] = 1e-12
            U = D / L[:, None]
            alive = np.ones(len(rows), bool)
            cell = search
            grid: dict[tuple, list] = {}
            for r in range(len(rows)):
                for P in (A[r], B[r]):
                    grid.setdefault((int(P[0] // cell), int(P[1] // cell)), []).append(r)
            merged_any = False
            for r1 in range(len(rows)):
                if not alive[r1]:
                    continue
                cs = set()
                for P in (A[r1], B[r1]):
                    gx, gy = int(P[0] // cell), int(P[1] // cell)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            cs.update(grid.get((gx + dx, gy + dy), ()))
                cs.discard(r1)
                if not cs:
                    continue
                c = np.fromiter(cs, int)
                c = c[alive[c]]
                if len(c) == 0:
                    continue
                u1 = U[r1]
                c = c[np.abs(U[c] @ u1) >= cos_tol]
                if len(c) == 0:
                    continue
                n1 = np.array([-u1[1], u1[0]])
                lat_a = np.abs((A[c] - A[r1]) @ n1)
                lat_b = np.abs((B[c] - A[r1]) @ n1)
                c = c[(lat_a <= 2 * tol) & (lat_b <= 2 * tol)]
                for r2 in c:
                    if self._try_geo_merge(rows[r1], rows[r2], ink, tol, search):
                        alive[r1] = alive[r2] = False
                        merged_any = True
                        break
            if not merged_any:
                return

    def _try_geo_merge(self, k1, k2, ink, tol, search):
        p1, p2 = self.prims[k1], self.prims[k2]
        A, B = self.V[p1.a], self.V[p1.b]
        C, D = self.V[p2.a], self.V[p2.b]
        d1 = B - A
        L1 = np.hypot(*d1)
        if L1 == 0:
            return False
        u1 = d1 / L1
        pts = {p1.a: A, p1.b: B, p2.a: C, p2.b: D}
        ids = list(pts)
        proj = [float(pts[i] @ u1) for i in ids]
        lo, hi = ids[int(np.argmin(proj))], ids[int(np.argmax(proj))]
        if lo == hi:
            return False
        E, F = pts[lo], pts[hi]
        ef = F - E
        Lef = np.hypot(*ef)
        if Lef == 0:
            return False
        nrm = np.array([-ef[1], ef[0]]) / Lef
        if any(abs(float((pts[i] - E) @ nrm)) > tol for i in ids):
            return False
        s1 = sorted([float(A @ u1), float(B @ u1)])
        s2 = sorted([float(C @ u1), float(D @ u1)])
        gap = max(s1[0], s2[0]) - min(s1[1], s2[1])
        if gap > search:
            return False
        if gap > 1.0:
            if s1[1] <= s2[0]:
                g0 = A if float(A @ u1) == s1[1] else B
                g1 = C if float(C @ u1) == s2[0] else D
            else:
                g0 = C if float(C @ u1) == s2[1] else D
                g1 = A if float(A @ u1) == s1[0] else B
            if not self._ink_between(ink, g0, g1):
                return False
        others = [i for i in ids if i not in (lo, hi)]
        p1.alive = p2.alive = False
        self.prims.append(Prim("line", lo, hi, interior=list(dict.fromkeys(
            others + p1.interior + p2.interior))))
        return True

    def merge_arcs_geometric(self, ink, tol):
        """Join arcs that lie on the same circle when the ink between them is
        continuous; promote to CIRCLE when they cover 360 degrees."""
        arcs = []
        for k, p in enumerate(self.prims):
            if p.alive and p.kind == "arc":
                ap = arc_params(self.V[p.a], p.mid, self.V[p.b])
                if ap is None:
                    continue
                a0 = math.atan2(self.V[p.a][1] - ap["cy"], self.V[p.a][0] - ap["cx"])
                if ap["sweep"] < 0:          # normalise to CCW interval
                    a0 = a0 + ap["sweep"]
                arcs.append(dict(k=k, cx=ap["cx"], cy=ap["cy"], r=ap["r"],
                                 a0=a0 % (2 * math.pi), sw=abs(ap["sweep"])))
        used = set()
        for i, s in enumerate(arcs):
            if s["k"] in used:
                continue
            grp = [s]
            for t in arcs[i + 1:]:
                if t["k"] in used:
                    continue
                if (math.hypot(s["cx"] - t["cx"], s["cy"] - t["cy"]) <= 1.5 * tol
                        and abs(s["r"] - t["r"]) <= tol):
                    grp.append(t)
            if len(grp) < 2:
                continue
            cx = float(np.mean([g["cx"] for g in grp]))
            cy = float(np.mean([g["cy"] for g in grp]))
            r = float(np.mean([g["r"] for g in grp]))
            grp.sort(key=lambda g: g["a0"])
            # angular intervals; merge neighbours across ink-covered gaps
            max_gap = min(math.radians(60), 60.0 / max(r, 1e-6))
            ivs = [[g["a0"], g["a0"] + g["sw"], [g["k"]]] for g in grp]
            merged = [ivs[0]]
            for iv in ivs[1:]:
                last = merged[-1]
                gap = iv[0] - last[1]
                if gap <= 0 or (gap <= max_gap and
                                self._ink_along_arc(ink, cx, cy, r, last[1], gap)):
                    last[1] = max(last[1], iv[1])
                    last[2] += iv[2]
                else:
                    merged.append(iv)
            # wrap-around
            max_gap = min(math.radians(60), 60.0 / max(r, 1e-6))
            if len(merged) > 1:
                first, last = merged[0], merged[-1]
                gap = first[0] + 2 * math.pi - last[1]
                if gap <= 0 or (gap <= max_gap and
                                self._ink_along_arc(ink, cx, cy, r, last[1], gap)):
                    last[1] = first[1] + 2 * math.pi
                    last[2] += first[2]
                    merged.pop(0)
            for a0, a1, ks in merged:
                if len(ks) < 2:
                    continue
                total = a1 - a0
                if total >= 2 * math.pi - 0.02 or (
                        len(merged) == 1 and 2 * math.pi - total <= max_gap
                        and self._ink_along_arc(ink, cx, cy, r, a1, 2 * math.pi - total)):
                    for k in ks:
                        self.prims[k].alive = False
                        used.add(k)
                    self.circles.append(Circle(cx, cy, r))
                    continue
                # build one arc; endpoints are the vertices nearest to a0/a1
                verts = set()
                for k in ks:
                    verts.update([self.prims[k].a, self.prims[k].b])

                def nearest(ang):
                    q = np.array([cx + r * math.cos(ang), cy + r * math.sin(ang)])
                    return min(verts, key=lambda v: float(np.hypot(*(self.V[v] - q))))
                va, vb = nearest(a0), nearest(a1)
                if va == vb:
                    continue
                am = (a0 + a1) / 2
                for k in ks:
                    self.prims[k].alive = False
                    used.add(k)
                self.prims.append(Prim("arc", va, vb, mid=np.array(
                    [cx + r * math.cos(am), cy + r * math.sin(am)])))

    def remove_covered_lines(self, tol):
        """Drop short lines lying entirely on a longer line (leftover
        bridges of merged crossings)."""
        lines = [k for k, p in enumerate(self.prims) if p.alive and p.kind == "line"]
        lines.sort(key=lambda k: -np.hypot(*(self.V[self.prims[k].a] - self.V[self.prims[k].b])))
        cell = 50.0
        grid: dict[tuple, list] = {}
        for k in lines:
            p = self.prims[k]
            A, B = self.V[p.a], self.V[p.b]
            n = int(np.hypot(*(B - A)) // cell) + 2
            for t in np.linspace(0, 1, n):
                q = A + (B - A) * t
                grid.setdefault((int(q[0] // cell), int(q[1] // cell)), []).append(k)
        for k in reversed(lines):        # shortest first
            p = self.prims[k]
            A, B = self.V[p.a], self.V[p.b]
            Lk = np.hypot(*(B - A))
            m = (A + B) / 2
            for k2 in set(grid.get((int(m[0] // cell), int(m[1] // cell)), ())):
                if k2 == k or not self.prims[k2].alive:
                    continue
                q = self.prims[k2]
                C, D = self.V[q.a], self.V[q.b]
                cd = D - C
                L = np.hypot(*cd)
                if L <= Lk:
                    continue
                u = cd / L
                nrm = np.array([-u[1], u[0]])
                ok = True
                for P in (A, B):
                    t = float((P - C) @ u)
                    if t < -tol or t > L + tol or abs(float((P - C) @ nrm)) > 2 * tol:
                        ok = False
                        break
                if ok:
                    p.alive = False
                    q.interior = list(dict.fromkeys(q.interior + [p.a, p.b]))
                    break

    # -- snapping ------------------------------------------------------------
    def snap_orthogonal(self, tol):
        for p in self.prims:
            if not p.alive or p.kind != "line":
                continue
            A, B = self.V[p.a], self.V[p.b]
            dx, dy = abs(A[0] - B[0]), abs(A[1] - B[1])
            if dy <= tol and dx > dy:
                y = (A[1] + B[1]) / 2
                A[1] = B[1] = y
            elif dx <= tol and dy > dx:
                x = (A[0] + B[0]) / 2
                A[0] = B[0] = x

    def project_interior(self):
        """Move vertices that lie on merged lines exactly onto them
        (T-junctions -> projection, X-crossings -> intersection)."""
        on: dict[int, list[int]] = {}
        for k, p in enumerate(self.prims):
            if p.alive and p.kind == "line":
                for q in p.interior:
                    on.setdefault(q, []).append(k)
        for q, ks in on.items():
            lines = []
            for k in ks:
                p = self.prims[k]
                A, B = self.V[p.a], self.V[p.b]
                d = B - A
                L = np.hypot(*d)
                if L > 0:
                    lines.append((A, d / L))
            if not lines:
                continue
            if len(lines) >= 2:
                (A1, d1), (A2, d2) = lines[0], lines[1]
                den = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(den) > 1e-6:
                    t = ((A2[0] - A1[0]) * d2[1] - (A2[1] - A1[1]) * d2[0]) / den
                    self.V[q][:] = A1 + t * d1
                    continue
            A, d = lines[0]
            self.V[q][:] = A + float((self.V[q] - A) @ d) * d

    # -- chain extraction (for polylines) ------------------------------------
    def chains(self):
        """Walk primitives through degree-2 vertices.

        Yields (vertex_ids, prim_indices, closed)."""
        inc = self.incidence()
        used = set()
        out = []

        def walk(v0, k0):
            vs = [v0]
            ks = []
            v, k = v0, k0
            while True:
                used.add(k)
                ks.append(k)
                p = self.prims[k]
                v = p.b if p.a == v else p.a
                vs.append(v)
                if v == v0:
                    return vs, ks, True
                nxt = [q for q in inc.get(v, []) if q not in used]
                if len(inc.get(v, [])) != 2 or not nxt:
                    return vs, ks, False
                k = nxt[0]

        for v, ks in inc.items():
            if len(ks) != 2:
                for k in ks:
                    if k not in used:
                        out.append(walk(v, k))
        for k, p in enumerate(self.prims):
            if p.alive and k not in used:
                out.append(walk(p.a, k))
        return out
