#pragma once
#include "Flux.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// Approximate Riemann solver — interface intentionally designed for replacement.
//
// Baseline choice: HLLE. The project deliberately prefers a well-validated
// HLLE solver over introducing HLLC too early:
//   * robust for strong shocks, with few pathological branches;
//   * extends naturally to GRHD and GRMHD;
//   * provides a clean reference for separating physical bugs from resolution
//     limits.
//
// Known tradeoff: HLLE is diffusive across contact discontinuities because it
// does not restore the intermediate wave. This is visible in shock tubes and
// is measured rather than treated as a bug.
//
// The interface consumes reconstructed left/right states and returns a face
// flux. A future HLLC/HLLD implementation can use the same signature.
// -----------------------------------------------------------------------------

struct RiemannInput {
    FluidVars      fL, fR;    // working variables on each side
    ConservedState uL, uR;    // corresponding conserved states
};

// HLLE numerical flux in direction dir.
//   a_L = min(0, lambda-_L, lambda-_R)
//   a_R = max(0, lambda+_L, lambda+_R)
//   F = (a_R F_L - a_L F_R + a_R a_L (U_R-U_L)) / (a_R-a_L)
// Supersonic cases return the upwind physical flux directly, guaranteeing
// F(U,U) = F_phys(U) exactly.
void hlleFlux(const RiemannInput& in, int dir, double F[5]);

}  // namespace grmhd
