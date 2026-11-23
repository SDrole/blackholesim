# Development Notes

This document records major architecture decisions, validation lessons and historical context for **blackholesim**.

For the forward-looking plan, see [ROADMAP.md](ROADMAP.md).

---

## Renderer evolution

### CPU rendering path

The original infinitesimally thin disk was replaced by a volumetric accretion-flow model with front-to-back transfer. The density model uses a configurable vertical scale height and the renderer integrates emission and attenuation through the flow.

A critical physics bug was found during this work: an explicit `g^4` brightness factor was being applied in addition to a temperature-shifted Planck treatment that already carried the relativistic boosting behavior. The duplicate factor produced over-bright, texture-poor images while remaining visually plausible.

The explicit duplicate factor was removed.

### Point-spread function

A Gaussian PSF/glow pass was added as an energy-conserving image-space effect rather than an arbitrary cinematic bloom.

---

## CUDA path

The project has a CUDA renderer for high-resolution workloads.

Major GPU-side additions have included:
- anisotropic MHD-inspired procedural turbulence;
- chromatic radiative-transfer opacity;
- higher-order photon-ring step refinement;
- adaptive supersampling near ray-classification boundaries.

### A100 validation lesson

The first real A100 run exposed a major luminance-calibration mismatch in the CUDA Planck pipeline. GPU output was roughly an order of magnitude too dim compared with the CPU convention even though the rendered image still looked qualitatively reasonable.

The fix introduced an explicit host-side Planck normalization consistent with the CPU path.

This reinforced a project-wide rule:

> CPU and CUDA implementations must be validated as separate numerical paths. Structural similarity is not evidence of numerical equivalence.

---

## Plasma interface

A shared `PlasmaState` abstraction was introduced so rendering no longer depends directly on one procedural disk implementation.

Conceptually:

```cpp
struct PlasmaState {
    double rho;
    double Theta_e;
    double u4[4];
    double B4[4];
    bool valid;
};
```

The interface is intended to support interchangeable sources:

```text
procedural plasma
public GRMHD snapshot
blackholesim GRMHD snapshot
        │
        ▼
   PlasmaState
        │
        ▼
radiative transfer
```

The CPU and CUDA paths must remain structurally aligned.

---

## GRMHD architecture decision

The project originally considered consuming established GRMHD snapshots only. The scope was later expanded to include a standalone in-house GRHD/GRMHD solver for research and validation purposes.

The renderer and solver remain separated through HDF5.

The practical reason is not only software architecture: independent components are easier to falsify. A renderer that directly shares internal state with its solver risks creating coupled errors that are difficult to isolate.

---

## Coordinate conventions

The ray tracer currently uses Boyer–Lindquist coordinates.

The fluid solver uses horizon-penetrating Kerr–Schild geometry and is designed around MKS/FMKS-style grids.

These are not interchangeable conventions. Coordinate transformations and primitive-variable meanings must be explicit at every snapshot boundary.

---

## Primitive-variable conventions

The hydro solver uses a HARM-inspired primitive velocity variable rather than storing ordinary coordinate three-velocity directly.

For hydro-only S2/S3, the Valencia conserved state is

```text
D   = rho W
S_i = rho h W^2 v_i
tau = rho h W^2 - p - D
```

The lapse and shift do not belong inside these local Eulerian definitions; they enter the curved-space fluxes, source terms and four-velocity mapping.

This distinction matters when extending the flat-space solver to S4.

---

## Methodological lessons from validation

### A test can be too good

An early connection test produced machine-precision agreement even though numerical metric differentiation should have imposed a much larger floor. The reason was algebraic self-confirmation: the test reused the same construction on both sides of the identity.

Lesson:

> A result substantially better than the theoretical numerical floor should be investigated, not automatically celebrated.

### Relative error can be ill-conditioned

Near the polar axis, one derivative tends to zero. Normalizing an absolute floating-point error by that vanishing derivative produced an apparent divergence in relative error even though the absolute error stayed nearly constant.

Lesson:

> Error normalization must use a scale that remains meaningful in the limit being tested.

### Convergence must be measured in the truncation regime

A first convergence study started with steps already small enough to reach the floating-point/differentiation floor. The inferred numerical order therefore oscillated and had no physical meaning.

Lesson:

> Measure formal order only where truncation error dominates the numerical floor.

### Exact oracles can be wrong

The SRHD exact Riemann implementation once returned residuals near `1e-16` for symmetric two-shock collisions while violating the exact symmetry condition `v* = 0`.

The error was a missing wave-family sign in the shock mass-flux relation.

Lesson:

> Solver convergence proves only that the implemented equations were solved. Independent physical symmetries and invariants are still required.

### Rollback must include diagnostics

A rejected SSPRK2 step previously restored physical state but retained diagnostic counters from the abandoned attempt.

The current contract separates:
- accepted-evolution counters;
- rejected-attempt telemetry.

This keeps both scientific accounting and failure diagnosis meaningful.

---

## Validation baseline

At the pre-S4 baseline `e52fbaf`:

| Suite | Result |
|---|---:|
| S0/S1 | 24 assertions, 0 failures |
| S1 consolidation | 12 assertions, 0 failures |
| S2 | 20 assertions / 145,036 sampled states, 0 failures |
| S3 | 91 assertions, 0 failures |
| **Total** | **147 assertions, 0 failures** |

Ray-tracer regression:
- 640×360 validation render;
- maximum Carter drift ≈ `1.95819e-07`;
- unchanged relative to the previous validated renderer baseline.

---

## Known limitations

- HLLE is intentionally diffusive across contact discontinuities.
- The exact SRHD Riemann oracle detects vacuum formation but does not yet sample the vacuum solution.
- The S3 HDF5 schema is intentionally minimal and must not be mistaken for a complete GRMHD snapshot.
- Production GRMHD floor/ceiling policies are not yet implemented.
- The renderer and CUDA path are still evolving and should not be treated as a stable public API.
- 2D GRMHD will be useful for development, but sustained MRI turbulence ultimately requires 3D.

---

## External comparison targets

The project plans to use:
- `ipole` for polarized radiative-transfer comparison;
- public iharm-style snapshots for input compatibility;
- Porth et al. (2019) metrics for GRMHD comparison;
- analytic Bondi/Michel and Fishbone–Moncrief solutions;
- Komissarov SRMHD shock tests;
- independent long-double and symmetry-based numerical oracles.

---

## Historical note

Some older internal planning documents contained decisions that were later superseded by the current roadmap. This file intentionally records the validated architectural lessons while [ROADMAP.md](ROADMAP.md) is the authoritative forward plan.
