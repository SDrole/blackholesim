#pragma once
#include <cmath>

namespace grmhd {

// -----------------------------------------------------------------------------
// Equation of state — isolated behind a minimal interface.
//
// Motivation: keep Gamma and thermodynamic relations out of the solver core.
// Anything that assumes an ideal gas goes through this class so a future EOS
// change (e.g. Mignone & McKinney / Synge) remains local.
//
// HARM convention: the primitive thermodynamic variable is INTERNAL-ENERGY
// DENSITY u, not pressure. epsilon = u/rho is the specific internal energy and
// h = 1 + epsilon + p/rho is the specific enthalpy.
//
// Gamma-law: p = (Gamma - 1) u.
// -----------------------------------------------------------------------------
class IdealGasEOS {
public:
    explicit IdealGasEOS(double gamma = 4.0 / 3.0) : gamma_(gamma) {}

    double gamma() const { return gamma_; }

    // p(rho,u) = (Gamma-1)u — independent of rho for a Gamma-law EOS.
    double pressure(double /*rho*/, double u) const { return (gamma_ - 1.0) * u; }

    // u(rho,p): inverse of pressure().
    double internalEnergy(double /*rho*/, double p) const { return p / (gamma_ - 1.0); }

    // Specific enthalpy h = 1 + epsilon + p/rho = 1 + Gamma u/rho.
    double specificEnthalpy(double rho, double u) const {
        return 1.0 + gamma_ * u / rho;
    }

    // Relativistic sound speed squared: cs^2 = Gamma p/(rho h).
    // Not needed in S2, but defined here so S3 does not duplicate EOS logic.
    double soundSpeed2(double rho, double u) const {
        const double p = pressure(rho, u);
        const double h = specificEnthalpy(rho, u);
        return (h > 0.0) ? (gamma_ * p / (rho * h)) : 0.0;
    }

private:
    double gamma_;
};

}  // namespace grmhd
