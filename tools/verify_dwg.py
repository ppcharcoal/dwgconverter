"""CI check: convert sample images to DWG and read the DWG back.

For every sample the DWG is decoded again (with LibreDWG's dwg2dxf and,
when installed, with ODA File Converter which reads DWG the way AutoCAD
does) and the entity counts are compared with what we wrote.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

import ezdxf
from ezdxf import recover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dwgconverter.export import (Scale, available_engines, build_doc,  # noqa: E402
                                 dxf_to_dwg_libredwg, dxf_to_dwg_oda, find_oda)
from dwgconverter.vectorize import Options, vectorize_file  # noqa: E402

GEOM = {"LINE", "ARC", "CIRCLE", "LWPOLYLINE", "HATCH"}


def counts_of(doc):
    return Counter(e.dxftype() for e in doc.modelspace() if e.dxftype() in GEOM)


class _Raw:
    """Minimal DXF reader: entity types in the ENTITIES section.  Used when
    a third-party DXF can't be loaded by ezdxf (e.g. handle quirks)."""

    def __init__(self, path):
        with open(path, errors="replace") as f:
            lines = [ln.strip() for ln in f]
        self.types = []
        in_ent = False
        for i in range(0, len(lines) - 1, 2):
            code, val = lines[i], lines[i + 1]
            if code == "2" and val == "ENTITIES":
                in_ent = True
            elif code == "0" and val == "ENDSEC":
                in_ent = False
            elif in_ent and code == "0":
                self.types.append(val)

    def modelspace(self):
        return [type("E", (), {"dxftype": (lambda self, t=t: t)})() for t in self.types]


def read_dxf(path):
    try:
        return ezdxf.readfile(path)
    except Exception:
        pass
    try:
        doc, _ = recover.readfile(path)
        return doc
    except Exception:
        return _Raw(path)


def oda_dwg_to_dxf(oda, dwg, out_dxf):
    with tempfile.TemporaryDirectory() as tin, tempfile.TemporaryDirectory() as tout:
        shutil.copy(dwg, os.path.join(tin, "x.dwg"))
        r = subprocess.run([oda, tin, tout, "ACAD2018", "DXF", "0", "1", "*.DWG"],
                           capture_output=True, text=True, timeout=300)
        res = os.path.join(tout, "x.dxf")
        errs = glob.glob(os.path.join(tout, "*.err"))
        err_txt = open(errs[0], errors="replace").read() if errs else ""
        if not os.path.isfile(res):
            raise RuntimeError("ODA could not read DWG: " + (r.stdout + r.stderr + err_txt).strip())
        shutil.copy(res, out_dxf)
        return err_txt


def main():
    samples = sys.argv[1:] or sorted(glob.glob("samples/*.png") + glob.glob("samples/*.jpg"))
    eng = available_engines()
    print("engines:", eng)
    lib = eng["libredwg"]
    dwg2dxf = None
    if lib:
        cand = os.path.join(os.path.dirname(lib), "dwg2dxf" + (".exe" if os.name == "nt" else ""))
        dwg2dxf = cand if os.path.isfile(cand) else shutil.which("dwg2dxf")
    oda = find_oda()
    out_dir = os.path.abspath("verify_out")
    os.makedirs(out_dir, exist_ok=True)
    failures = 0
    for s in samples:
        name = os.path.splitext(os.path.basename(s))[0]
        res = vectorize_file(s, Options())
        doc = build_doc(res, Scale())
        want = counts_of(doc)
        print(f"\n== {name}: wrote {dict(want)}")
        targets = []
        if lib:
            dxf = os.path.join(out_dir, name + "_r2000.dxf")
            build_doc(res, Scale(), "R2000").saveas(dxf)
            dwg = os.path.join(out_dir, name + "_libredwg.dwg")
            try:
                dxf_to_dwg_libredwg(dxf, dwg, lib)
                targets.append(("libredwg", dwg))
            except Exception as ex:
                print("  libredwg write FAILED:", ex)
                failures += 1
        if oda:
            dxf = os.path.join(out_dir, name + "_r2010.dxf")
            doc.saveas(dxf)
            dwg = os.path.join(out_dir, name + "_oda.dwg")
            try:
                dxf_to_dwg_oda(dxf, dwg, "2018", oda)
                targets.append(("oda", dwg))
            except Exception as ex:
                print("  oda write FAILED:", ex)
                failures += 1
        for writer, dwg in targets:
            print(f"  [{writer}] {os.path.basename(dwg)} {os.path.getsize(dwg)} bytes")
            if dwg2dxf:
                back = dwg + ".libredwg_back.dxf"
                r = subprocess.run([dwg2dxf, "-y", "-o", back, dwg], capture_output=True, text=True)
                if os.path.isfile(back):
                    got = counts_of(read_dxf(back))
                    print(f"    read by LibreDWG : {dict(got)} {'OK' if got == want else 'MISMATCH'}")
                else:
                    print("    read by LibreDWG : FAILED", (r.stdout + r.stderr)[-300:])
            if oda:
                back = dwg + ".oda_back.dxf"
                try:
                    err = oda_dwg_to_dxf(oda, dwg, back)
                    got = counts_of(read_dxf(back))
                    ok = got == want
                    print(f"    read by ODA      : {dict(got)} {'OK' if ok else 'MISMATCH'}")
                    if err.strip():
                        print("    ODA audit log:", err.strip()[:500])
                    if not ok:
                        failures += 1
                except Exception as ex:
                    print("    read by ODA      : FAILED", ex)
                    failures += 1
    print("\nfailures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
