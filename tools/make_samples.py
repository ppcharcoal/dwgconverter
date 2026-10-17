"""Generate sample drawings used for testing (samples/*.png)."""
import os
import sys

import cv2
import numpy as np

AA = cv2.LINE_AA
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "samples")
os.makedirs(out, exist_ok=True)

# 1) floor plan
img = np.full((1400, 2000), 255, np.uint8)
for (x0, y0, x1, y1) in [(100, 100, 1900, 100), (100, 100, 100, 1300), (100, 1300, 1900, 1300),
                         (1900, 100, 1900, 1300), (900, 100, 900, 800), (100, 800, 1900, 800),
                         (1400, 800, 1400, 1300)]:
    cv2.line(img, (x0, y0), (x1, y1), 0, 3, AA)
    if y0 == y1:
        cv2.line(img, (x0, y0 + 14), (x1, y1 + 14), 0, 3, AA)
    else:
        cv2.line(img, (x0 + 14, y0), (x1 + 14, y1), 0, 3, AA)
cv2.ellipse(img, (914, 500), (150, 150), 0, 0, 90, 0, 2, AA)
cv2.line(img, (914, 500), (1064, 500), 0, 2, AA)
cv2.ellipse(img, (400, 800), (120, 120), 0, 180, 270, 0, 2, AA)
for c in [(300, 300), (1600, 300), (300, 1100)]:
    cv2.circle(img, c, 40, 0, 2, AA)
cv2.circle(img, (1650, 1050), 150, 0, 3, AA)
cv2.line(img, (100, 1360), (1900, 1360), 0, 1, AA)
for x in (100, 900, 1900):
    cv2.line(img, (x - 8, 1368), (x + 8, 1352), 0, 1, AA)
cv2.putText(img, "8000", (450, 1350), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, AA)
for x in range(1000, 1850, 40):
    cv2.line(img, (x, 600), (x + 20, 600), 0, 2, AA)
cv2.rectangle(img, (880, 780), (944, 844), 0, -1)
cv2.imwrite(os.path.join(out, "floor_plan.png"), img)
cv2.imwrite(os.path.join(out, "floor_plan_scan.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 70])

# 2) mechanical part
img = np.full((900, 1200), 255, np.uint8)
pts = np.array([[150, 150], [900, 150], [1050, 300], [1050, 750], [150, 750]], np.int32)
cv2.polylines(img, [pts], True, 0, 3, AA)
cv2.circle(img, (400, 450), 140, 0, 3, AA)
cv2.circle(img, (400, 450), 60, 0, 2, AA)
for c in [(800, 300), (800, 600), (950, 450)]:
    cv2.circle(img, c, 30, 0, 2, AA)
cv2.line(img, (230, 450), (570, 450), 0, 1, AA)      # centre lines
cv2.line(img, (400, 280), (400, 620), 0, 1, AA)
cv2.ellipse(img, (650, 750), (100, 100), 0, 180, 360, 0, 3, AA)  # cut-out
cv2.rectangle(img, (180, 650), (330, 720), 0, -1)                 # filled block
cv2.imwrite(os.path.join(out, "mechanical_part.png"), img)
print("samples written to", os.path.abspath(out))
