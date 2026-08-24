# blackholesim — Simulateur de trou noir de Kerr

Rendu d'un trou noir **en rotation** (métrique de Kerr) par tracé de
**géodésiques nulles**, avec disque d'accrétion relativiste de **Novikov-Thorne**.
Au programme : lentille gravitationnelle, ombre asymétrique (entraînement des
référentiels / frame-dragging), anneau de photons, décalage spectral complet
(gravitationnel + Doppler + frame-dragging) et vérification de la **constante de
Carter**. Sortie : une image **et** des matrices de données scientifiques.

Unités géométrisées `G = c = 1`. La masse fixe l'échelle (`M = 1`), le spin est
`a/M ∈ [0, 1)`. `a = 0` redonne Schwarzschild.

## Physique

**Métrique de Kerr** (Boyer-Lindquist), `Σ = r² + a²cos²θ`, `Δ = r² - 2Mr + a²`.

**Géodésiques — formalisme de Hamilton-Jacobi.** On intègre les équations de
Hamilton issues du super-hamiltonien `H = ½ g^{μν} p_μ p_ν = 0` :

```
dxⁱ/dλ = ∂H/∂pᵢ ,     dpᵢ/dλ = -∂H/∂xⁱ
```

`t` et `φ` étant cycliques, `E = -p_t` et `Lz = p_φ` sont conservés par
construction. On ne propage que `(r, θ, φ, p_r, p_θ)`. Avantage décisif : les
points de rebroussement (`p_r`, `p_θ` passent par 0) sont traversés **sans
ambiguïté de signe** `√R`, contrairement aux équations séparées du 1er ordre.
Les dérivées `∂H/∂r`, `∂H/∂θ` sont calculées analytiquement.

**Constante de Carter.** `Q = p_θ² + cos²θ(Lz²/sin²θ - a²E²)`, et la constante du
tenseur de Killing `K = Q + (Lz - aE)²`. Elle est recalculée à chaque pas et sa
dérive relative est suivie : le programme vérifie `max|ΔK|/|K| < 10⁻⁶`
(typiquement ~10⁻⁷) et exporte une carte par pixel.

**Évitement de singularité.** Les coordonnées de Boyer-Lindquist ont des
singularités de coordonnées en `r₊` (`Δ→0`) et sur l'axe (`sinθ→0`). On les
traite par : (1) arrêt analytique du rayon à `r₊·(1+ε)` — le photon est capturé,
donc `Δ` n'atteint jamais 0 (aucune division par zéro) ; (2) pas borné par
`0.3(r-r₊)` près de l'horizon ; (3) planchers sur `Δ` et `sinθ`.

**Disque de Novikov-Thorne.** Profil de flux relativiste de **Page-Thorne (1974)**
à partir de l'ISCO de Kerr (formule de Bardeen). Température `T(r) ∝ F(r)^{1/4}`.
Décalage observé `g = 1/[uᵗ(1 - Ω Lz)]` (orbite circulaire prograde,
`Ω = √M/(r^{3/2}+a√M)`), couleur de corps noir décalée en `g·T`, brillance
amplifiée `∝ g⁴`.

## Architecture (`src/`)

| Fichier | Rôle |
|---|---|
| `Config.hpp`   | Tous les paramètres (spin, caméra, disque, tolérances) |
| `Vec3`         | Algèbre vectorielle 3D (couleurs, direction de fond) |
| `Metric`       | Géométrie de Kerr, ISCO, orbites, redshift, flux Page-Thorne |
| `Geodesic`     | État canonique, équations de Hamilton, constante de Carter |
| `Integrator`   | RK4 (fixe) et RKF45 (adaptatif, contrôle de pas pondéré) |
| `Camera`       | Plan image (α,β) + conditions initiales de Bardeen |
| `Renderer`     | Tracé arrière, disque N.-Thorne, suivi de Carter, ombrage |
| `Image`        | Tone-mapping + écriture PPM / BMP, export matrices CSV |
| `main.cpp`     | Assemblage, CLI, rapport de conservation de Carter |

## Compilation (Windows / MSVC)

**Prérequis** (une seule fois) : Visual Studio Build Tools (workload C++), CMake, Ninja.
```powershell
winget install --id Microsoft.VisualStudio.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Kitware.CMake -e
winget install --id Ninja-build.Ninja -e
```

**Build** — utiliser `build_release.bat` à la racine (fiable : ajoute `vswhere.exe`
au `PATH` avant d'appeler `vcvars64.bat`, sans quoi l'environnement MSVC ne
s'initialise pas correctement) :
```powershell
.\build_release.bat
```
Le rendu est **CPU pur** (OpenMP multi-thread) sur une machine sans GPU NVIDIA —
le CUDA ci-dessous ne s'active que si `nvcc` est détecté.

## Exécution

```powershell
# largeur hauteur AA  spin(a/M)  dossier_sortie
./build/blackholesim.exe 1200 675 2 0.9 output
./build/blackholesim.exe 800 450 1 0.0 output   # cas Schwarzschild (a=0)
```

## Sorties (`output/`)

- `blackhole_kerr.bmp` / `.ppm` — image rendue
- `redshift.csv` — facteur `g` du disque par pixel (`0` ailleurs)
- `hit_type.csv` — `0` = fond, `1` = disque, `2` = horizon
- `carter_drift.csv` — dérive relative de `K` le long de chaque rayon (vérification)

## Pistes d'extension

Disque géométriquement épais, orbites rétrogrades, observateur ZAMO à distance
finie (tétrade exacte), coordonnées de Kerr-Schild pénétrant l'horizon,
intégration spectrale par bande, polarisation.
