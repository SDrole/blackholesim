#pragma once
#include "Fluid.hpp"
#include "EOS.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// S3 — SRHD physical fluxes and characteristic speeds.
//
// Minkowski spacetime here: alpha=1, beta^i=0, gamma_ij=delta_ij. The
// interface is written in terms of local fluid variables so S4 can generalize
// it to curved spacetime without replacing the entire solver contract.
//
// Perfect-fluid stress-energy tensor:
//   T^{mu nu} = rho h u^mu u^nu + p g^{mu nu}
//
// S2 Valencia conserved variables:
//   D   = rho W
//   S_i = rho h W^2 v_i
//   tau = rho h W^2 - p - D
//
// Flux in direction x:
//   F_D     = D v^x
//   F_{S_i} = S_i v^x + p delta^x_i
//   F_tau   = (tau+p) v^x = S^x - D v^x
//
// At rest, D=rho, S_i=0 and tau=u. Every flux vanishes except the normal
// momentum flux, which is exactly the pressure.
// -----------------------------------------------------------------------------

// Working primitive variables used by the flux calculation. Fluxes naturally
// use v^i rather than u_tilde^i. The conversion is local: v^i=u_tilde^i/W.
struct FluidVars {
    double rho = 0.0;
    double u   = 0.0;      // internal-energy density
    double v[3] = {0.0, 0.0, 0.0};
    double p   = 0.0;
    double W   = 1.0;
    double h   = 1.0;      // specific enthalpy
    double cs2 = 0.0;      // sound speed squared
};

// Build working variables from S2 primitives.
FluidVars makeFluidVars(const PrimitiveState& P, const IdealGasEOS& eos);

// Physical flux in direction dir (0=x,1=y,2=z).
// Component order: (D,S_x,S_y,S_z,tau).
void physicalFlux(const FluidVars& f, const ConservedState& U, int dir,
                  double F[5]);

// -----------------------------------------------------------------------------
// RELATIVISTIC CHARACTERISTIC SPEEDS
//
// v +/- c_s is wrong in relativity. Velocity composition is non-Galilean and,
// with transverse motion, eigenvalues depend on the full v^2 rather than only
// the normal component.
//
// For 1D SRHD in Valencia form:
//   lambda_0 = v^x
//   lambda_+/- =
//     [v^x(1-c_s^2) +/- c_s sqrt((1-v^2)
//        [1-v^2 c_s^2-(v^x)^2(1-c_s^2)])]
//     / (1-v^2 c_s^2)
//
// Validation checks include:
//   * v -> 0: lambda_+/- -> +/-c_s
//   * c_s -> 0: lambda_+/- -> v^x
//   * zero transverse velocity: relativistic velocity-addition identity
//   * |lambda|<1 across a broad physical-state sweep
// -----------------------------------------------------------------------------
struct SignalSpeeds {
    double lambdaMinus = 0.0;
    double lambdaPlus  = 0.0;
    double lambda0     = 0.0;
};

SignalSpeeds characteristicSpeeds(const FluidVars& f, int dir);

// Numerical-guard instrumentation. These clamps are useful as final protection
// in release code, but validation must demonstrate that scientific tests do
// not pass because the guards modified the physics.
struct CharacteristicClampCounters {
    long long cs2Clamped      = 0;  // cs^2 clamped to [0,1]
    long long discNegative    = 0;  // negative radicand -> zero root
    long long denomDegenerate = 0;  // 1-v^2 cs^2 ~ 0
    long long lambdaClamped   = 0;  // |lambda| clamped to <=1
    long long total() const {
        return cs2Clamped + discNegative + denomDegenerate + lambdaClamped;
    }
};
const CharacteristicClampCounters& characteristicClampCounters();
void resetCharacteristicClampCounters();

}  // namespace grmhd
