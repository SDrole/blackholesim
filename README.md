# blackholesim

<div align="center">

**A physics-first Kerr black hole renderer and an evolving GRHD/GRMHD research codebase**

[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/CPU-OpenMP-1f6feb)](https://www.openmp.org/)
![Status](https://img.shields.io/badge/status-active%20research-orange)

Null-geodesic ray tracing · Kerr spacetime · Novikov–Thorne accretion disk · Carter-constant diagnostics · SRHD validation · GRMHD roadmap

</div>

---

## Overview

**blackholesim** is a numerical black hole simulation and rendering project built around one rule:

> A result is not considered correct because it looks plausible. It must survive independent physical and numerical checks.

The current renderer traces **null geodesics in the Kerr metric**, models a relativistic **Novikov–Thorne accretion disk**, computes gravitational and Doppler redshift, resolves frame dragging and higher-order photon-ring structure, and exports both rendered images and scientific diagnostic matrices.

In parallel, the repository contains a deliberately staged relativistic hydrodynamics codebase in `grmhd/`. The long-term goal is a self-consistent GRMHD pipeline whose snapshots can feed the renderer and polarized radiative-transfer modules.

The project uses geometrized units `G = c = 1`, with `M = 1` setting the length and time scale.

> [!IMPORTANT]
> This is an active research/development project, not a production astrophysics package. Numerical claims are tied to explicit tests and should be interpreted within the validated domain documented below.

---

## Highlights

- **Kerr spacetime** in Boyer–Lindquist coordinates for ray tracing.
- **Hamiltonian null-geodesic integration** with analytic metric derivatives.
- **Carter constant monitoring** as a per-ray numerical invariant.
- **Novikov–Thorne / Page–Thorne disk physics** with relativistic orbital motion.
- **Gravitational + Doppler + frame-dragging redshift**.
- **CPU renderer** with OpenMP and an optional **CUDA** path.
- **Higher-order photon-ring refinement** and adaptive supersampling.
- **Volumetric accretion-flow model** with front-to-back radiative transfer.
- Scientific outputs in addition to images: redshift, hit classification, Carter drift.
- Independent `grmhd/` development track with Kerr–Schild geometry, SRHD primitive recovery, HLLE, PLM reconstruction and SSPRK2.
- Explicit failure policies, floor diagnostics and adversarial tests designed to detect self-confirming validation.

---

## Physics

### Kerr geometry

The renderer uses the Kerr metric in Boyer–Lindquist coordinates,

```text
Σ = r² + a² cos²θ
Δ = r² - 2Mr + a²
```

with dimensionless spin `a/M`. The Schwarzschild limit is recovered at `a = 0`.

### Null geodesics

Photon trajectories are integrated in Hamiltonian form from

```text
H = 1/2 g^{μν} p_μ p_ν = 0
```

using

```text
dxⁱ/dλ =  ∂H/∂pᵢ
dpᵢ/dλ = -∂H/∂xⁱ
```

Because `t` and `φ` are cyclic coordinates, `E = -p_t` and `Lz = p_φ` are conserved by construction. The code evolves `(r, θ, φ, p_r, p_θ)`, which allows radial and polar turning points to be crossed without manually choosing the sign of a square root.

### Carter constant

For every ray, the renderer monitors

```text
Q = p_θ² + cos²θ (Lz² / sin²θ - a²E²)
K = Q + (Lz - aE)²
```

and tracks the relative drift of `K` throughout integration.

The current regression configuration reports a maximum Carter drift of approximately **1.96 × 10⁻⁷**, below the project threshold of `10⁻⁶`.

### Accretion disk

The thermal disk model follows the relativistic Novikov–Thorne / Page–Thorne framework, with the Kerr ISCO and a temperature profile derived from the relativistic flux.

For prograde circular motion,

```text
Ω = √M / (r^(3/2) + a√M)
```

and the observed redshift factor is evaluated consistently with the emitter four-velocity and photon constants of motion.

---

## Repository structure

```text
blackholesim/
├── src/                     Kerr ray tracer and renderer
│   ├── Metric.*             Kerr geometry, ISCO, orbital quantities
│   ├── Geodesic.*           Canonical photon state and Hamilton equations
│   ├── Integrator.*         RK4 / adaptive integration infrastructure
│   ├── Camera.*             Observer and image-plane initial conditions
│   ├── Renderer.*           CPU ray tracing and radiative transfer
│   ├── Image.*              Image output, tone mapping and scientific exports
│   ├── Plasma.hpp           Shared plasma-state interface
│   └── cuda/Renderer.cu     CUDA rendering path
│
├── grmhd/                   Relativistic hydro / future GRMHD solver track
│   ├── KerrSchild.*         Horizon-penetrating Kerr–Schild geometry
│   ├── Coordinates.hpp      MKS coordinate mapping
│   ├── Fluid.*              Primitive ↔ conserved SRHD state conversion
│   ├── Flux.*               Physical fluxes and characteristic speeds
│   ├── Riemann.*            HLLE numerical flux
│   ├── ExactRiemannSRHD.*   Independent exact SRHD Riemann oracle
│   ├── Solver1D.*           Finite-volume SRHD solver
│   └── tests_s*.cpp         Scientific and numerical validation suites
│
├── docs/
│   ├── ROADMAP.md           Scientific roadmap and validation strategy
│   └── DEVELOPMENT.md       Architecture decisions and development history
│
├── CMakeLists.txt
├── build_release.bat
└── setup_a100.sh
```

---

## Current validation status

The pre-GRHD baseline at commit `e52fbaf` reports:

| Validation group | Result |
|---|---:|
| S0 / S1 geometry & infrastructure | 24 assertions, 0 failures |
| S1 consolidation | 12 assertions, 0 failures |
| S2 primitive/conserved recovery | 20 assertions, 145,036 sampled states, 0 failures |
| S3 SRHD solver | 91 assertions, 0 failures |
| **Total** | **147 assertions, 0 failures** |
| Ray-tracer regression | Carter drift ≈ **1.95819 × 10⁻⁷** |

The test philosophy intentionally includes:
- analytic invariants;
- convergence studies;
- independent reference formulations;
- symmetric counterexamples;
- forced recovery failures;
- rollback tests;
- checks that numerical clamps remain inactive during validation;
- explicit rejection of unsupported physical regimes instead of silently fabricating a solution.

A recent hardening pass also caught a subtle sign error in the exact two-shock SRHD oracle: a symmetric head-on collision produced an apparently well-converged but physically impossible non-zero contact velocity. The test suite now contains symmetry checks specifically designed to prevent that class of self-confirming error.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full validation plan.

---

## Build

### Requirements

- C++20-capable compiler
- CMake
- Ninja (recommended)
- OpenMP
- Optional: CUDA toolkit / `nvcc`
- Optional for the GRMHD snapshot path: HDF5

### Windows / MSVC

Install the toolchain once:

```powershell
winget install --id Microsoft.VisualStudio.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Kitware.CMake -e
winget install --id Ninja-build.Ninja -e
```

Then build with:

```powershell
.\build_release.bat
```

The helper script initializes the MSVC environment and configures an optimized build. CUDA is enabled only when a compatible CUDA toolchain is detected.

### Generic CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

---

## Running the renderer

```powershell
# width height AA spin(a/M) output_directory
./build/blackholesim.exe 1200 675 2 0.9 output

# Schwarzschild limit
./build/blackholesim.exe 800 450 1 0.0 output
```

Typical scientific outputs include:

| File | Meaning |
|---|---|
| `blackhole_kerr.bmp` / `.ppm` | Rendered image |
| `redshift.csv` | Disk redshift factor `g` per pixel |
| `hit_type.csv` | Ray classification: background / disk / horizon |
| `carter_drift.csv` | Relative Carter-constant drift per ray |

---

## GRHD / GRMHD development track

The hydrodynamics solver is intentionally developed separately from the renderer:

```text
[ GRHD / GRMHD solver ]
          │
          ▼
   iharm-compatible HDF5
          │
          ▼
[ ray tracer + radiative transfer ]
```

This separation keeps the solver independently testable and allows public/reference GRMHD snapshots to be consumed by the renderer.

Current staged roadmap:

- **S0** — infrastructure and output
- **S1** — Kerr–Schild geometry and MKS coordinates
- **S2** — relativistic hydro primitive ↔ conserved recovery
- **S3** — finite-volume SRHD + HLLE + PLM + SSPRK2
- **S4** — GRHD in Kerr–Schild spacetime
- **S5** — magnetic GRMHD, induction and constrained transport
- **S6** — production floors, ceilings and boundary conditions
- **S7** — magnetized torus, MRI and community-code comparison

S0–S3 are the established baseline. S4 is the next major physics milestone.

---

## Scientific roadmap

Five realism tracks are planned in parallel:

1. Time variability and interstellar scattering.
2. Electron-temperature prescriptions (`R_high/R_low`).
3. Physical 230 GHz synchrotron emission.
4. Full Stokes `I,Q,U,V` polarized transfer and Faraday effects.
5. Self-consistent GRMHD evolution.

The project aims to validate polarized radiative transfer against established tools such as **ipole** and GRMHD behavior against analytic tests and published code-comparison benchmarks.

Full details: **[Scientific roadmap →](docs/ROADMAP.md)**

---

## Design principles

1. **Fail loudly rather than return a plausible physical lie.**
2. **Use independent oracles whenever possible.**
3. **Measure convergence above the floating-point noise floor.**
4. **Instrument every floor, fallback and numerical guard.**
5. **Keep accepted-state diagnostics separate from rejected-step telemetry.**
6. **Do not use visual similarity as scientific validation.**
7. **Keep the solver and renderer decoupled through a documented data format.**

---

## References

Core references currently used by the project include:

- Page & Thorne (1974), relativistic thin-disk flux.
- Bardeen, Press & Teukolsky, Kerr circular orbits and ISCO.
- Noble et al. (2006), GRMHD primitive recovery.
- Tóth (2000), constrained transport.
- Porth et al. (2019), EHT GRMHD code comparison.
- Pandya et al. (2016), polarized synchrotron coefficients.
- Mościbrodzka & Gammie (2018), polarized radiative transfer.
- EHT polarized radiative-transfer code comparison (2023).

See the roadmap for links and implementation context.

---

## Contributing

The project is evolving quickly and scientific correctness takes priority over feature count.

Before proposing a physics or numerical change, please read **[CONTRIBUTING.md](CONTRIBUTING.md)**. New numerical methods should normally include a discriminating test, an invariant, an independent oracle, or a convergence study.

---

## Project status

**Active development.** The renderer is functional, while the GRHD/GRMHD subsystem is being built milestone-by-milestone.

No release or API-stability guarantee is currently provided.
