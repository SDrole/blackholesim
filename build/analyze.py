import csv, os
from collections import Counter

OUT = r"C:\Users\toute\Desktop\blackholesim\output"

def load(name, cast):
    with open(os.path.join(OUT, name)) as f:
        return [[cast(v) for v in row] for row in csv.reader(f)]

hit = load("hit_type.csv", int)
red = load("redshift.csv", float)
car = load("carter_drift.csv", float)
H = len(hit); W = len(hit[0])

c = Counter(v for row in hit for v in row)
print("Resolution        : %dx%d (%d pixels)" % (W, H, W * H))
print("Fond (0)          :", c.get(0, 0))
print("Disque (1)        :", c.get(1, 0))
print("Horizon/ombre (2) :", c.get(2, 0))

xs = [i for j in range(H) for i in range(W) if hit[j][i] == 2]
ys = [j for j in range(H) for i in range(W) if hit[j][i] == 2]
if xs:
    cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
    print("Centroide ombre   : x=%.1f (centre %.1f, decal=%.1f px)  y=%.1f"
          % (cx, W/2, cx - W/2, cy))

gl, gr = [], []
for j in range(H):
    for i in range(W):
        if hit[j][i] == 1:
            (gl if i < W // 2 else gr).append(red[j][i])

def stats(a):
    return (min(a), sum(a)/len(a), max(a)) if a else (0.0, 0.0, 0.0)

print("g disque gauche   : min=%.3f moy=%.3f max=%.3f (n=%d)" % (stats(gl) + (len(gl),)))
print("g disque droite   : min=%.3f moy=%.3f max=%.3f (n=%d)" % (stats(gr) + (len(gr),)))

flat = [v for row in car for v in row]
cmax = max(flat); cmean = sum(flat)/len(flat)
over = sum(1 for v in flat if v > 1e-6)
print("Carter drive K    : max=%.3e  moy=%.3e  pixels>1e-6: %d / %d"
      % (cmax, cmean, over, len(flat)))
print("-> Carter K conservee (<1e-6):", "OUI" if cmax < 1e-6 else "NON (%d pixels)" % over)
