"""Accuracy tests: one stroke must become one entity, at the right place."""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dwgconverter.export import Scale, build_doc  # noqa: E402
from dwgconverter.raster import binarize  # noqa: E402
from dwgconverter.vectorize import Options, vectorize_image_mask  # noqa: E402

AA = cv2.LINE_AA


def run(img, **kw):
    return vectorize_image_mask(binarize(img), Options(**kw))


def blank(h=400, w=600):
    return np.full((h, w), 255, np.uint8)


def kinds(res):
    out = {}
    for e in res.entities:
        out[e[0]] = out.get(e[0], 0) + 1
    return out


def lines(res):
    return [(np.array(e[1]), np.array(e[2])) for e in res.entities if e[0] == "line"]


def near(p, q, tol=2.0):
    return float(np.hypot(*(np.asarray(p, float) - np.asarray(q, float)))) <= tol


def has_line(res, a, b, tol=2.0):
    return any((near(p, a, tol) and near(q, b, tol)) or (near(p, b, tol) and near(q, a, tol))
               for p, q in lines(res))


@pytest.mark.parametrize("thick", [1, 2, 3, 5])
def test_single_line_is_one_entity(thick):
    img = blank()
    cv2.line(img, (40, 50), (560, 330), 0, thick, AA)
    res = run(img)
    assert kinds(res) == {"line": 1}
    assert has_line(res, (40, 50), (560, 330), 2.5)


def test_horizontal_and_vertical_exact():
    img = blank()
    cv2.line(img, (50, 100), (550, 100), 0, 2)
    cv2.line(img, (300, 150), (300, 380), 0, 2)
    res = run(img)
    assert kinds(res) == {"line": 2}
    for p, q in lines(res):
        assert abs(p[0] - q[0]) < 1e-9 or abs(p[1] - q[1]) < 1e-9  # snapped orthogonal


def test_rectangle_is_four_lines():
    img = blank()
    cv2.rectangle(img, (100, 80), (500, 320), 0, 2)
    res = run(img)
    assert kinds(res) == {"line": 4}
    for c in [(100, 80), (500, 80), (500, 320), (100, 320)]:
        assert sum(near(p, c, 2) + near(q, c, 2) for p, q in lines(res)) == 2


def test_crossing_lines_stay_whole():
    img = blank()
    cv2.line(img, (50, 50), (550, 350), 0, 2, AA)
    cv2.line(img, (50, 350), (550, 50), 0, 2, AA)
    cv2.line(img, (300, 20), (300, 380), 0, 2, AA)
    res = run(img)
    assert kinds(res) == {"line": 3}


def test_t_junction():
    img = blank()
    cv2.line(img, (50, 100), (550, 100), 0, 3)
    cv2.line(img, (300, 100), (300, 350), 0, 3)
    res = run(img)
    assert kinds(res) == {"line": 2}
    assert has_line(res, (50, 100), (550, 100), 2.5)


def test_circle():
    img = blank()
    cv2.circle(img, (300, 200), 120, 0, 2, AA)
    res = run(img)
    assert kinds(res) == {"circle": 1}
    c = [e for e in res.entities if e[0] == "circle"][0]
    assert near(c[1], (300, 200), 1.0)
    assert abs(c[2] - 120) < 1.0


def test_circle_with_line_through_it():
    img = blank()
    cv2.circle(img, (300, 200), 120, 0, 2, AA)
    cv2.line(img, (100, 200), (500, 200), 0, 2, AA)
    res = run(img)
    assert kinds(res) == {"circle": 1, "line": 1}


def test_quarter_arc():
    img = blank()
    cv2.ellipse(img, (200, 300), (200, 200), 0, 270, 360, 0, 2, AA)
    res = run(img)
    assert kinds(res) == {"arc": 1}


def test_dashed_line_keeps_gaps():
    img = blank()
    for x in range(50, 550, 50):
        cv2.line(img, (x, 200), (x + 25, 200), 0, 2)
    res = run(img)
    assert kinds(res) == {"line": 10}


def test_filled_rectangle_outline_exact():
    img = blank()
    img[100:200, 150:400] = 0   # pixels x 150..399, y 100..199
    res = run(img)
    assert kinds(res) == {"poly": 1}
    poly = [e for e in res.entities if e[0] == "poly"][0]
    pts = np.array(poly[1])
    assert len(pts) == 4
    # outline runs along pixel edges: x 149.5..399.5 in pixel-centre coords
    assert np.allclose(sorted(set(np.round(pts[:, 0], 3))), [149.5, 399.5], atol=0.05)
    assert np.allclose(sorted(set(np.round(pts[:, 1], 3))), [99.5, 199.5], atol=0.05)


def test_dxf_coordinates_one_to_one():
    img = blank(400, 600)
    cv2.line(img, (100, 50), (500, 50), 0, 1)
    res = run(img)
    doc = build_doc(res, Scale())
    (ln,) = list(doc.modelspace().query("LINE"))
    ys = {round(ln.dxf.start.y, 3), round(ln.dxf.end.y, 3)}
    xs = sorted([ln.dxf.start.x, ln.dxf.end.x])
    assert ys == {400 - 50.5}            # image row 50 -> y = H - 50.5
    assert abs(xs[0] - 100.5) < 1 and abs(xs[1] - 500.5) < 1


def test_mm_scale():
    img = blank(400, 600)
    cv2.circle(img, (300, 200), 100, 0, 2, AA)
    res = run(img)
    doc = build_doc(res, Scale.from_dpi(254, "mm"))   # 0.1 mm per pixel
    (c,) = list(doc.modelspace().query("CIRCLE"))
    assert abs(c.dxf.radius - 10.0) < 0.1
    assert doc.header["$INSUNITS"] == 4


def test_arc_direction_in_dxf():
    img = blank()
    cv2.ellipse(img, (300, 200), (150, 150), 0, 180, 360, 0, 2, AA)   # upper half in image
    res = run(img)
    doc = build_doc(res, Scale())
    (a,) = list(doc.modelspace().query("ARC"))
    # upper half of the image is the upper half in CAD too (y up), i.e. 0..180 deg
    mid = math.radians((a.dxf.start_angle + ((a.dxf.end_angle - a.dxf.start_angle) % 360) / 2))
    assert math.sin(mid) > 0.9
