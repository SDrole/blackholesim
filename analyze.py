import csv
import os
import sys
from collections import Counter

OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
)


def load(name, cast):
    with open(os.path.join(OUT, name)) as f:
        return [[cast(v) for v in row] for row in csv.reader(f)]


hit = load("hit_type.csv", int)
red = load("redshift.csv", float)
car = load("carter_drift.csv", float)
H = len(hit)
W = len(hit[0])

c = Counter(v for row in hit for v in row)
print("Resolution        : %dx%d (%d pixels)" % (W, H, W * H))
print("Background (0)    :", c.get(0, 0))
print("Disk (1)          :", c.get(1, 0))
print("Horizon/shadow (2):", c.get(2, 0))

xs = [i for j in range(H) for i in range(W) if hit[j][i] == 2]
ys = [j for j in range(H) for i in range(W) if hit[j][i] == 2]
if xs:
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    print(
        "Shadow centroid   : x=%.1f (center %.1f, offset=%.1f px)  y=%.1f"
        % (cx, W / 2, cx - W / 2, cy)
    )

gl, gr = [], []
for j in range(H):
    for i in range(W):
        if hit[j][i] == 1:
            (gl if i < W // 2 else gr).append(red[j][i])


def stats(a):
    return (min(a), sum(a) / len(a), max(a)) if a else (0.0, 0.0, 0.0)


print(
    "g disk left       : min=%.3f mean=%.3f max=%.3f (n=%d)"
    % (stats(gl) + (len(gl),))
)
print(
    "g disk right      : min=%.3f mean=%.3f max=%.3f (n=%d)"
    % (stats(gr) + (len(gr),))
)

flat = [v for row in car for v in row]
cmax = max(flat)
cmean = sum(flat) / len(flat)
over = sum(1 for v in flat if v > 1e-6)
print(
    "Carter drift K    : max=%.3e mean=%.3e pixels>1e-6: %d / %d"
    % (cmax, cmean, over, len(flat))
)
print(
    "-> Carter K conserved (<1e-6):",
    "YES" if cmax < 1e-6 else "NO (%d pixels)" % over,
)
