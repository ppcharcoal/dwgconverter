# DWG Converter – JPG/PNG to DWG/DXF

Windows program that turns raster drawings (JPG, PNG, BMP, TIFF) into real
CAD geometry that opens in **AutoCAD, ZWCAD, BricsCAD, DraftSight, LibreCAD…**

* **1 : 1 geometry** – the drawing overlays the source image exactly
  (1 pixel = 1 drawing unit by default, or real millimetres/inches from the
  image DPI).
* **One stroke = one entity** – a straight line becomes a single `LINE`, a
  circle a single `CIRCLE`, an arc a single `ARC` – never hundreds of tiny
  segments. Lines interrupted by crossings, T-junctions or other strokes are
  re-joined; dashed lines stay dashed.
* Filled areas (solid blocks, thick walls, bold text) are traced as exact
  closed outlines, optionally solid-hatched.
* Batch conversion, preview with zoom/pan, command-line tool for scripting.

## Download

Get the latest build from the **Releases → “Latest build”** page (or the
*Actions* tab → newest run → artifact `DWGConverter-windows-x64`).
Unzip anywhere and run **`DWGConverter.exe`** – no installation needed.

### DWG engines

DXF files are written directly. For DWG the program uses:

| Engine | Notes |
| --- | --- |
| **ODA File Converter** (recommended) | Free download from [opendesign.com](https://www.opendesign.com/guestfiles/oda_file_converter). Install it once; the program finds it automatically. Produces DWG 2000–2018 that AutoCAD/ZWCAD open cleanly. |
| **LibreDWG** (bundled) | Works out of the box, writes DWG 2000 format. Used when ODA isn't installed. |

If a DWG ever gives trouble, tick *Also keep a DXF copy* – every CAD program
opens DXF.

## Using the app

1. **Add images…** (or drag image files onto `DWGConverter.exe`).
2. Pick options (defaults work for most drawings):
   * **Mode** – `auto` traces thin strokes as centre lines and filled areas as
     outlines; `centerline` / `outline` force one method.
   * **Tolerance** – how far (in pixels) the result may deviate from the
     source strokes. Higher = fewer, longer entities; lower = more detail.
   * **Max line width** – in auto mode, anything thicker is outlined.
   * **Units** – *Pixels* gives a 1:1 pixel drawing. Choose millimetres/inches
     to scale by the image DPI (or type the DPI the drawing was scanned at).
3. **Preview selected** to check the result (red = lines, blue = arcs and
   circles, green = outlines), then **Convert all**.

## Command line

```
dwgconvert.exe drawing.png                       # -> drawing.dwg
dwgconvert.exe scans\*.jpg -o out --units mm --dpi 300
dwgconvert.exe part.png -f dxf --tolerance 1.0 --polylines
dwgconvert.exe --engines                         # show DWG engines found
dwgconvert.exe --help
```

## How it works

1. The image is binarised (Otsu threshold) and specks are removed.
2. Solid areas are separated from thin strokes with a morphological opening.
3. Thin strokes are skeletonised and traced into a graph of strokes and
   junctions; skeleton spurs and junction artefacts are cleaned up.
4. Every stroke is split into the *fewest* straight lines and circular arcs
   that stay within the tolerance (greedy longest fit with line/circle
   least-squares fits); corners rounded by the skeleton are restored.
5. Junction positions are solved as the intersection of the strokes meeting
   there, then collinear pieces and co-circular arcs are merged across
   junctions and crossings (only where the image has continuous ink between
   them), near-horizontal/vertical lines are snapped.
6. Filled areas are outlined with sub-pixel marching squares on the pixel
   edges and fitted the same way into closed polylines with arc segments.

## Building from source

```
pip install -r requirements.txt pyinstaller
python -m pytest tests
pyinstaller dwgconverter.spec
```

The GitHub Actions workflow (`.github/workflows/build.yml`) builds the
Windows package, bundles LibreDWG, converts the files in `samples/` and
reads the resulting DWGs back (with LibreDWG and ODA File Converter) to check
that every entity survived.

## Limitations

* Text is converted to geometry (strokes/outlines), not to editable `TEXT`.
* Photos or shaded images are not drawings; results will be noisy.
* Very small details (a few pixels) are approximated within the tolerance.

## Licenses

This program: MIT (see `LICENSE`). The bundled LibreDWG tools are GPLv3
(see `libredwg/COPYING.txt`, source at https://github.com/LibreDWG/libredwg);
they are run as separate programs. ODA File Converter is not bundled.
