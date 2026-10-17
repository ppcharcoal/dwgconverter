"""Fitting of straight lines and circular arcs to dense pixel chains.

A traced stroke is a dense list of points (one per pixel).  We split it
greedily into the *fewest* primitives that stay within ``tol`` of every
source point, so a straight stroke becomes exactly one line and a curved
stroke becomes a handful of arcs instead of hundreds of tiny segments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Seg:
    """A fitted primitive covering points[i0..i1] (inclusive) of a chain."""
    kind: str               # "line" or "arc"
    i0: int
    i1: int
    # line: direction / anchor of the TLS fit; arc: center + radius
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0
    p: np.ndarray | None = None   # line anchor point
    d: np.ndarray | None = None   # line unit direction
    corner: np.ndarray | None = None  # forced joint with the next segment


# --------------------------------------------------------------------------
# Elementary fits
# --------------------------------------------------------------------------
def fit_line(P: np.ndarray):
    """Total-least-squares line.  Returns (anchor, unit_dir, max_deviation)."""
    c = P.mean(axis=0)
    Q = P - c
    if len(P) < 2:
        return c, np.array([1.0, 0.0]), 0.0
    # 2x2 covariance eigen-decomposition (faster than SVD on N x 2)
    sxx = float((Q[:, 0] * Q[:, 0]).sum())
    syy = float((Q[:, 1] * Q[:, 1]).sum())
    sxy = float((Q[:, 0] * Q[:, 1]).sum())
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    d = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-d[1], d[0]])
    dev = float(np.abs(Q @ n).max())
    return c, d, dev


def fit_circle(P: np.ndarray, iters: int = 4):
    """Algebraic (Kasa) circle fit refined with a few Gauss-Newton steps.

    Returns (cx, cy, r, max_radial_deviation) or None if degenerate.
    """
    n = len(P)
    if n < 3:
        return None
    x = P[:, 0]
    y = P[:, 1]
    mx = x.sum() / n
    my = y.sum() / n
    u = x - mx
    v = y - my
    # Kasa normal equations
    suu = float(u @ u); svv = float(v @ v); suv = float(u @ v)
    su = float(u.sum()); sv = float(v.sum())
    z = u * u + v * v
    M = np.array([[suu, suv, su], [suv, svv, sv], [su, sv, float(n)]])
    rhs = -np.array([float(u @ z), float(v @ z), float(z.sum())])
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -sol[0] / 2.0, -sol[1] / 2.0
    r2 = cx * cx + cy * cy - sol[2]
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(r2)
    if r > 1e7:
        return None
    for _ in range(iters):
        dx = u - cx
        dy = v - cy
        dist = np.sqrt(dx * dx + dy * dy)
        dist[dist == 0] = 1e-12
        res = dist - r
        jx = -dx / dist
        jy = -dy / dist
        JTJ = np.array([[jx @ jx, jx @ jy, -jx.sum()],
                        [jx @ jy, jy @ jy, -jy.sum()],
                        [-jx.sum(), -jy.sum(), float(n)]])
        JTr = np.array([jx @ res, jy @ res, -res.sum()])
        try:
            delta = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            break
        cx += delta[0]
        cy += delta[1]
        r += delta[2]
        if not np.isfinite(r) or r <= 0:
            return None
        if abs(delta).max() < 1e-4:
            break
    dist = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
    dev = float(np.abs(dist - r).max())
    return cx + mx, cy + my, r, dev


def _arc_sweep(P: np.ndarray, cx: float, cy: float) -> float:
    """Signed sweep angle (radians) travelled along P around (cx, cy).
    Returns NaN if the travel direction is not consistent."""
    ang = np.unwrap(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    dif = np.diff(ang)
    total = float(ang[-1] - ang[0])
    if total == 0:
        return float("nan")
    # pixel staircase may cause tiny backwards steps; reject real reversals
    back = dif[np.sign(dif) != np.sign(total)]
    if len(back) and np.abs(back).sum() > 0.15 * abs(total) + 0.05:
        return float("nan")
    return total


# --------------------------------------------------------------------------
# Greedy segmentation
# --------------------------------------------------------------------------
class ChainFitter:
    def __init__(self, tol: float = 1.5, min_arc_points: int = 8,
                 min_arc_sweep_deg: float = 12.0, max_radius: float = 1e6,
                 allow_arcs: bool = True):
        self.tol = tol
        self.min_arc_points = min_arc_points
        self.min_arc_sweep = math.radians(min_arc_sweep_deg)
        self.max_radius = max_radius
        self.allow_arcs = allow_arcs
        self.ink = None          # optional bool mask, enables corner recovery
        self.dist = None         # optional distance transform of ink

    # -- acceptance tests ---------------------------------------------------
    def line_ok(self, P):
        _, _, dev = fit_line(P)
        return dev <= self.tol

    def arc_fits(self, P):
        """Lenient test used while extending: circle within tolerance."""
        if len(P) < self.min_arc_points:
            return None
        res = fit_circle(P)
        if res is None:
            # collinear points: a "circle" of infinite radius fits a line
            _, _, dev = fit_line(P)
            return (0.0, 0.0, float("inf")) if dev <= self.tol else None
        cx, cy, r, dev = res
        ok = dev <= self.tol and r >= self.tol
        if ok and r < 1e5:
            sweep = _arc_sweep(P, cx, cy)
            ok = np.isfinite(sweep) and abs(sweep) <= 2 * math.pi - 1e-3
        if ok:
            return cx, cy, r
        # a short, nearly straight window is compatible with a large arc
        _, _, ldev = fit_line(P)
        return (0.0, 0.0, float("inf")) if ldev <= self.tol else None

    def arc_ok(self, P):
        if len(P) < self.min_arc_points:
            return None
        res = fit_circle(P)
        if res is None:
            return None
        cx, cy, r, dev = res
        if dev > self.tol or r > self.max_radius or r < self.tol:
            return None
        sweep = _arc_sweep(P, cx, cy)
        if not np.isfinite(sweep):
            return None
        if abs(sweep) < self.min_arc_sweep or abs(sweep) > 2 * math.pi - 1e-3:
            return None
        # an arc must actually be curved beyond the tolerance, otherwise
        # it is a line in disguise
        sagitta = r * (1 - math.cos(abs(sweep) / 2.0))
        if sagitta <= self.tol:
            return None
        return cx, cy, r

    # -- extension search ---------------------------------------------------
    def _extend(self, P, i, test, start_len):
        """Largest j such that test(P[i:j+1]) holds (galloping + bisection).
        Returns (j, result) or (None, None)."""
        n = len(P)
        j0 = i + start_len - 1
        if j0 > n - 1:
            return None, None
        r0 = test(P[i:j0 + 1])
        if not r0:
            return None, None
        good, good_res = j0, r0
        step = max(2, start_len)
        bad = None
        while True:
            j = min(good + step, n - 1)
            if j == good:
                break
            r = test(P[i:j + 1])
            if r:
                good, good_res = j, r
                step *= 2
                if j == n - 1:
                    break
            else:
                bad = j
                break
        if bad is not None:
            lo, hi = good, bad
            while hi - lo > 1:
                mid = (lo + hi) // 2
                r = test(P[i:mid + 1])
                if r:
                    lo, good_res = mid, r
                else:
                    hi = mid
            good = lo
        return good, good_res

    def segment(self, P: np.ndarray) -> list[Seg]:
        """Split an open chain into lines/arcs."""
        n = len(P)
        segs: list[Seg] = []
        if n < 2:
            return segs
        i = 0
        while i < n - 1:
            jl, _ = self._extend(P, i, self.line_ok, 2)
            if jl is None:
                jl = i + 1
            ja, ares = (None, None)
            if self.allow_arcs and n - i >= self.min_arc_points:
                ja, _ = self._extend(P, i, self.arc_fits, self.min_arc_points)
                if ja is not None and ja > jl + 1:
                    ares = self.arc_ok(P[i:ja + 1])
                    if not ares:
                        ja = None
            if ja is not None and ja > jl + 1 and ares:
                cx, cy, r = ares
                segs.append(Seg("arc", i, ja, cx=cx, cy=cy, r=r))
                i = ja
            else:
                c, d, _ = fit_line(P[i:jl + 1])
                segs.append(Seg("line", i, jl, p=c, d=d))
                i = jl
        segs = self._merge_adjacent(P, segs)
        if self.ink is not None:
            segs = self._recover_corners(P, segs)
        return segs

    def _ink_at(self, q):
        x, y = int(round(q[0])), int(round(q[1]))
        H, W = self.ink.shape
        return 0 <= x < W and 0 <= y < H and bool(self.ink[y, x])

    def _recover_corners(self, P, segs):
        """Skeletons round off sharp corners.  An 'arc' whose two tangent
        lines meet at a point that is still covered by ink is really a
        corner (a true fillet leaves that point empty)."""
        out = []
        for s in segs:
            if s.kind != "arc":
                out.append(s)
                continue
            Q = P[s.i0:s.i1 + 1]
            n = len(Q)
            chord = Q[-1] - Q[0]
            L = np.hypot(*chord)
            if L == 0 or n < 7:
                out.append(s)
                continue
            nrm = np.array([-chord[1], chord[0]]) / L
            k = int(np.argmax(np.abs((Q - Q[0]) @ nrm)))
            if k < 3 or k > n - 4:
                out.append(s)
                continue
            m = max(1, int(0.2 * min(k, n - 1 - k)))
            A = Q[:k - m + 1]
            B = Q[k + m:]
            if len(A) < 3 or len(B) < 3:
                out.append(s)
                continue
            ca, da, deva = fit_line(A)
            cb, db, devb = fit_line(B)
            den = da[0] * db[1] - da[1] * db[0]
            if deva > self.tol or devb > self.tol or abs(den) < math.sin(math.radians(45)):
                out.append(s)
                continue
            t = ((cb[0] - ca[0]) * db[1] - (cb[1] - ca[1]) * db[0]) / den
            X = ca + t * da
            off = X - Q[k]
            dX = np.hypot(*off)
            if dX > max(3 * self.tol, 0.35 * L) or not self._ink_at(X):
                out.append(s)
                continue
            # a real corner keeps ink beyond X (the stroke's outer corner);
            # a curve's outer edge is already behind X
            if dX > 0.5 and self.dist is not None:
                h = max(1.0, float(self.dist[int(round(Q[k][1])), int(round(Q[k][0]))]))
                if not self._ink_at(X + off / dX * 0.7 * h):
                    out.append(s)
                    continue
            # the corner must be outside the chord side where the arc bulges
            out.append(Seg("line", s.i0, s.i0 + k, p=ca, d=da, corner=X))
            out.append(Seg("line", s.i0 + k, s.i1, p=cb, d=db))
        return out

    def _merge_adjacent(self, P, segs):
        """Greedy left-to-right may split one curve in two; re-join when the
        union still fits."""
        changed = True
        while changed and len(segs) > 1:
            changed = False
            out = [segs[0]]
            for s in segs[1:]:
                a = out[-1]
                Q = P[a.i0:s.i1 + 1]
                if a.kind == "line" and s.kind == "line":
                    c, d, dev = fit_line(Q)
                    if dev <= self.tol:
                        out[-1] = Seg("line", a.i0, s.i1, p=c, d=d)
                        changed = True
                        continue
                elif a.kind == "arc" and s.kind == "arc":
                    res = self.arc_ok(Q)
                    if res:
                        cx, cy, r = res
                        out[-1] = Seg("arc", a.i0, s.i1, cx=cx, cy=cy, r=r)
                        changed = True
                        continue
                out.append(s)
            segs = out
        return segs

    # -- closed loops ---------------------------------------------------------
    def full_circle(self, P: np.ndarray):
        """If a closed loop is a circle, return (cx, cy, r)."""
        if len(P) < max(self.min_arc_points, 8):
            return None
        res = fit_circle(P)
        if res is None:
            return None
        cx, cy, r, dev = res
        if dev > self.tol or r < self.tol:
            return None
        sweep = _arc_sweep(np.vstack([P, P[:1]]), cx, cy)
        if not np.isfinite(sweep) or abs(abs(sweep) - 2 * math.pi) > 0.3:
            return None
        return cx, cy, r


def best_loop_start(P: np.ndarray, tol: float) -> int:
    """Index of the sharpest corner of a closed loop (a good split point)."""
    n = len(P)
    if n < 5:
        return 0
    k = max(2, int(round(3 * tol)) + 1)
    k = min(k, n // 3) or 1
    a = P[(np.arange(n) - k) % n]
    b = P[(np.arange(n) + k) % n]
    v1 = P - a
    v2 = b - P
    n1 = np.hypot(v1[:, 0], v1[:, 1]) + 1e-12
    n2 = np.hypot(v2[:, 0], v2[:, 1]) + 1e-12
    cosang = (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]) / (n1 * n2)
    return int(np.argmin(cosang))
