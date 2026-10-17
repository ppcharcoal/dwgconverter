"""Command line interface: dwgconvert image.png [-o out.dwg] [options]"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from . import __version__
from .export import Scale, available_engines, save
from .vectorize import Options, vectorize_file


def build_parser():
    ap = argparse.ArgumentParser(
        prog="dwgconvert",
        description="Convert JPG/PNG drawings into DWG/DXF with real lines, arcs and circles.")
    ap.add_argument("inputs", nargs="*", help="image files (wildcards allowed)")
    ap.add_argument("-o", "--output", help="output file (single input) or folder")
    ap.add_argument("-f", "--format", choices=["dwg", "dxf"], default="dwg")
    ap.add_argument("--mode", choices=["auto", "centerline", "outline"], default="auto",
                    help="auto: thin strokes -> centerlines, filled areas -> outlines")
    ap.add_argument("-t", "--tolerance", type=float, default=1.5,
                    help="max distance (pixels) between result and source strokes")
    ap.add_argument("--threshold", type=int, default=None, help="0-255, default automatic")
    ap.add_argument("--ink", choices=["auto", "dark", "light"], default="auto")
    ap.add_argument("--denoise", action="store_true", help="smooth JPEG noise first")
    ap.add_argument("--min-speck", type=int, default=6, help="ignore blobs smaller than N px")
    ap.add_argument("--max-line-width", type=float, default=12.0,
                    help="auto mode: thicker areas are traced as outlines")
    ap.add_argument("--no-arcs", action="store_true", help="only straight lines")
    ap.add_argument("--no-snap", action="store_true", help="don't snap near-horizontal/vertical lines")
    ap.add_argument("--polylines", action="store_true", help="join connected segments into polylines")
    ap.add_argument("--fill", action="store_true", help="solid-hatch filled areas")
    ap.add_argument("--units", choices=["px", "mm", "cm", "in", "m"], default="px",
                    help="px = 1 drawing unit per pixel (default)")
    ap.add_argument("--dpi", type=float, default=None,
                    help="DPI for unit conversion (default: from image, else 300)")
    ap.add_argument("--scale", type=float, default=None,
                    help="explicit drawing units per pixel (overrides --dpi)")
    ap.add_argument("--engine", choices=["auto", "oda", "libredwg"], default="auto")
    ap.add_argument("--dwg-version", choices=["2000", "2004", "2007", "2010", "2013", "2018"],
                    default="2018", help="DWG version (ODA engine only)")
    ap.add_argument("--frame", action="store_true", help="add the image border rectangle")
    ap.add_argument("--keep-dxf", action="store_true", help="also save a DXF next to the DWG")
    ap.add_argument("--engines", action="store_true", help="show available DWG engines and exit")
    ap.add_argument("--version", action="version", version=__version__)
    return ap


def make_scale(args, dpi_from_image):
    if args.units == "px":
        return Scale(args.scale or 1.0, "unitless")
    if args.scale:
        return Scale(args.scale, args.units)
    return Scale.from_dpi(args.dpi or dpi_from_image or 300.0, args.units)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.engines:
        for k, v in available_engines().items():
            print(f"{k:9s}: {v or 'not found'}")
        return 0
    files = []
    for pat in args.inputs:
        hits = glob.glob(pat)
        files += hits if hits else [pat]
    if not files:
        build_parser().print_help()
        return 2
    opts = Options(mode=args.mode, tolerance=args.tolerance, threshold=args.threshold,
                   ink=args.ink, denoise=args.denoise, min_speck=args.min_speck,
                   max_line_width=args.max_line_width, detect_arcs=not args.no_arcs,
                   snap_orthogonal=not args.no_snap, join_polylines=args.polylines,
                   fill_outlines=args.fill)
    rc = 0
    for f in files:
        try:
            if args.output and (len(files) > 1 or os.path.isdir(args.output)):
                os.makedirs(args.output, exist_ok=True)
                out = os.path.join(args.output, os.path.splitext(os.path.basename(f))[0])
            elif args.output:
                out = args.output
            else:
                out = os.path.splitext(f)[0]
            res = vectorize_file(f, opts)
            info = save(res, out + ("" if out.lower().endswith((".dwg", ".dxf")) else ".x"),
                        make_scale(args, res.dpi), args.format, args.engine,
                        args.dwg_version, args.frame, args.keep_dxf)
            st = ", ".join(f"{v} {k}" for k, v in res.stats.items() if k != "seconds")
            print(f"{f} -> {', '.join(info['files'])}  [{info['engine']}]  {st}")
        except Exception as ex:
            print(f"ERROR {f}: {ex}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
