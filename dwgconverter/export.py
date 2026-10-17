"""Writing vector results to DXF and DWG.

DXF is written directly with ezdxf.  DWG is produced from that DXF by an
external converter:

* ODA File Converter (free download from opendesign.com) -- preferred, gives
  DWG files that AutoCAD / ZWCAD open without complaints.  Auto-detected in
  "C:\\Program Files\\ODA\\..." or via the DWGCONV_ODA environment variable.
* LibreDWG ``dwgwrite`` (bundled with the Windows build) -- fallback, writes
  AutoCAD 2000 format DWG.
"""
from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import ezdxf
import numpy as np

from .geometry import arc_params

# DXF $INSUNITS codes
UNITS = {"unitless": 0, "in": 1, "ft": 2, "mm": 4, "cm": 5, "m": 6}

LAYERS = {
    "CENTERLINES": 7,   # white/black
    "OUTLINES": 7,
    "FILL": 8,          # grey
    "IMAGE_FRAME": 9,
}


@dataclass
class Scale:
    """Pixel -> drawing unit mapping.  factor = drawing units per pixel."""
    factor: float = 1.0
    units: str = "unitless"

    @staticmethod
    def from_dpi(dpi: float, units: str = "mm") -> "Scale":
        per_inch = {"mm": 25.4, "cm": 2.54, "m": 0.0254, "in": 1.0}[units]
        return Scale(per_inch / dpi, units)


class Transform:
    """Image pixel coords (x right, y down, pixel centers at integers)
    -> CAD coords (y up).  The image occupies [0, W] x [0, H] * factor,
    so the drawing overlays the picture 1:1."""

    def __init__(self, height: int, scale: Scale):
        self.h = height
        self.f = scale.factor

    def __call__(self, p):
        return ((float(p[0]) + 0.5) * self.f, (self.h - float(p[1]) - 0.5) * self.f)


def _bulge(pa, pm, pb) -> float:
    ap = arc_params(pa, pm, pb)
    if ap is None:
        return 0.0
    return math.tan(ap["sweep"] / 4.0)


def _poly_points(ent, T):
    """LWPOLYLINE points (x, y, bulge) from a ('poly', pts, mids, closed)."""
    _, pts, mids, closed, _layer = ent
    P = [T(p) for p in pts]
    out = []
    n = len(P)
    nseg = n if closed else n - 1
    for i in range(n):
        bulge = 0.0
        if i < nseg and mids[i] is not None:
            bulge = _bulge(P[i], T(mids[i]), P[(i + 1) % n])
        out.append((P[i][0], P[i][1], bulge))
    return out


def build_doc(result, scale: Scale, version: str = "R2010", frame: bool = False):
    doc = ezdxf.new(version, setup=False)
    doc.header["$INSUNITS"] = UNITS.get(scale.units, 0)
    doc.header["$MEASUREMENT"] = 0 if scale.units in ("in", "ft") else 1
    for name, color in LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name, color=color)
    msp = doc.modelspace()
    T = Transform(result.height, scale)

    def add_loop_to_hatch(h, ent):
        if ent[0] == "circle":
            c = T(ent[1])
            r = ent[2] * scale.factor
            ep = h.paths.add_edge_path()
            ep.add_arc(c, r, 0, 360)
        elif ent[0] == "poly":
            pts = _poly_points(ent, T)
            h.paths.add_polyline_path(pts, is_closed=True)

    for e in result.entities:
        kind = e[0]
        if kind == "line":
            msp.add_line(T(e[1]), T(e[2]), dxfattribs={"layer": e[3]})
        elif kind == "arc":
            pa, pm, pb = T(e[1]), T(e[2]), T(e[3])
            ap = arc_params(pa, pm, pb)
            if ap is None:
                msp.add_line(pa, pb, dxfattribs={"layer": e[4]})
                continue
            a0 = math.degrees(math.atan2(pa[1] - ap["cy"], pa[0] - ap["cx"]))
            a1 = math.degrees(math.atan2(pb[1] - ap["cy"], pb[0] - ap["cx"]))
            if not ap["ccw"]:
                a0, a1 = a1, a0
            msp.add_arc((ap["cx"], ap["cy"]), ap["r"], a0, a1, dxfattribs={"layer": e[4]})
        elif kind == "circle":
            msp.add_circle(T(e[1]), e[2] * scale.factor, dxfattribs={"layer": e[3]})
        elif kind == "poly":
            msp.add_lwpolyline(_poly_points(e, T), format="xyb", close=e[3],
                               dxfattribs={"layer": e[4]})
        elif kind == "hatch":
            h = msp.add_hatch(color=256, dxfattribs={"layer": e[2]})
            h.set_solid_fill(color=256)
            for loop in e[1]:
                add_loop_to_hatch(h, loop)

    W = result.width * scale.factor
    H = result.height * scale.factor
    if frame:
        msp.add_lwpolyline([(0, 0), (W, 0), (W, H), (0, H)], close=True,
                           dxfattribs={"layer": "IMAGE_FRAME"})
    doc.header["$EXTMIN"] = (0, 0, 0)
    doc.header["$EXTMAX"] = (W, H, 0)
    doc.header["$LIMMIN"] = (0, 0)
    doc.header["$LIMMAX"] = (W, H)
    try:
        doc.set_modelspace_vport(height=H * 1.05, center=(W / 2, H / 2))
    except Exception:
        pass
    return doc


def write_dxf(result, path: str, scale: Scale, version: str = "R2010", frame: bool = False):
    doc = build_doc(result, scale, version, frame)
    doc.saveas(path)
    return path


# --------------------------------------------------------------------------
# DWG engines
# --------------------------------------------------------------------------
def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_oda() -> str | None:
    env = os.environ.get("DWGCONV_ODA")
    if env and os.path.isfile(env):
        return env
    if os.name != "nt":
        p = shutil.which("ODAFileConverter")
        return p
    cands = []
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        cands += glob.glob(os.path.join(root, "ODA", "ODAFileConverter*", "ODAFileConverter.exe"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def find_libredwg() -> str | None:
    exe = "dwgwrite.exe" if os.name == "nt" else "dwgwrite"
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    for d in (os.path.join(_app_dir(), "libredwg"), os.path.join(exe_dir, "libredwg"),
              _app_dir(), exe_dir):
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return shutil.which("dwgwrite")


def available_engines() -> dict:
    return {"oda": find_oda(), "libredwg": find_libredwg()}


def _run(cmd, timeout=600):
    kw = {}
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kw["startupinfo"] = si
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


ODA_VERSIONS = {"2000": "ACAD2000", "2004": "ACAD2004", "2007": "ACAD2007",
                "2010": "ACAD2010", "2013": "ACAD2013", "2018": "ACAD2018"}


def dxf_to_dwg_oda(dxf_path: str, dwg_path: str, version: str = "2018", oda: str | None = None):
    oda = oda or find_oda()
    if not oda:
        raise RuntimeError("ODA File Converter not found")
    with tempfile.TemporaryDirectory() as tin, tempfile.TemporaryDirectory() as tout:
        name = "drawing.dxf"
        shutil.copy(dxf_path, os.path.join(tin, name))
        cmd = [oda, tin, tout, ODA_VERSIONS.get(version, "ACAD2018"), "DWG", "0", "1", "*.DXF"]
        r = _run(cmd)
        out = os.path.join(tout, "drawing.dwg")
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            msg = (r.stdout + r.stderr).strip()
            errs = glob.glob(os.path.join(tout, "*.err"))
            if errs:
                with open(errs[0], errors="replace") as f:
                    msg += "\n" + f.read()
            raise RuntimeError("ODA File Converter failed: " + (msg or f"exit code {r.returncode}"))
        shutil.move(out, dwg_path)
    return dwg_path


def dxf_to_dwg_libredwg(dxf_path: str, dwg_path: str, exe: str | None = None):
    exe = exe or find_libredwg()
    if not exe:
        raise RuntimeError("LibreDWG (dwgwrite) not found")
    if os.path.exists(dwg_path):
        os.remove(dwg_path)
    r = _run([exe, "-y", "--as", "r2000", "-o", dwg_path, dxf_path])
    if not os.path.isfile(dwg_path) or os.path.getsize(dwg_path) == 0:
        raise RuntimeError("LibreDWG failed: " + ((r.stdout + r.stderr).strip()[-800:]
                                                    or f"exit code {r.returncode}"))
    return dwg_path


def save(result, out_path: str, scale: Scale, fmt: str = "dwg", engine: str = "auto",
         dwg_version: str = "2018", frame: bool = False, keep_dxf: bool = False) -> dict:
    """Save `result` as DXF or DWG.  Returns info dict {files, engine}."""
    out_path = os.path.abspath(out_path)
    base, _ = os.path.splitext(out_path)
    if fmt == "dxf":
        write_dxf(result, base + ".dxf", scale, "R2010", frame)
        return {"files": [base + ".dxf"], "engine": "ezdxf"}

    engines = available_engines()
    order = {"auto": ["oda", "libredwg"], "oda": ["oda"], "libredwg": ["libredwg"]}[engine]
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        for eng in order:
            if not engines.get(eng):
                errors.append(f"{eng}: not installed")
                continue
            try:
                if eng == "oda":
                    dxf = os.path.join(tmp, "oda.dxf")
                    write_dxf(result, dxf, scale, "R2010", frame)
                    dxf_to_dwg_oda(dxf, base + ".dwg", dwg_version, engines["oda"])
                else:
                    dxf = os.path.join(tmp, "lib.dxf")
                    write_dxf(result, dxf, scale, "R2000", frame)
                    dxf_to_dwg_libredwg(dxf, base + ".dwg", engines["libredwg"])
                files = [base + ".dwg"]
                if keep_dxf:
                    write_dxf(result, base + ".dxf", scale, "R2010", frame)
                    files.append(base + ".dxf")
                return {"files": files, "engine": eng}
            except Exception as ex:  # try the next engine
                errors.append(f"{eng}: {ex}")
    raise RuntimeError("Could not create DWG.\n" + "\n".join(errors) +
                       "\nInstall the free ODA File Converter or save as DXF instead.")
