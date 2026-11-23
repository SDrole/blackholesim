#pragma once

namespace bhsim {

// -----------------------------------------------------------------------------
// Local plasma state at one spacetime point. This is the shared interface
// between the current procedural backend and a future GRMHD snapshot backend.
//
// All quantities use geometrized units (G=c=1), consistent with the rest of
// the renderer. A GRMHD backend will populate the same fields by interpolating
// snapshot data; downstream synchrotron and polarized-transfer code depends
// only on this structure, not on the data source.
// -----------------------------------------------------------------------------
struct PlasmaState {
    double rho     = 0.0;   // density; currently used as a relative opacity factor
    double Theta_e = 0.0;   // dimensionless electron temperature kT_e/(m_e c^2)
                             // populated once an electron-temperature model is enabled

    // Contravariant fluid four-velocity in local Boyer-Lindquist coordinates:
    // u^t, u^r, u^theta, u^phi.
    double u4[4] = {0.0, 0.0, 0.0, 0.0};

    // Contravariant magnetic four-vector in the same basis. In the current
    // procedural backend this is a toy toroidal field, not a physical GRMHD
    // solution. It becomes physically meaningful with snapshot-driven plasma.
    double B4[4] = {0.0, 0.0, 0.0, 0.0};

    bool valid = false;   // false outside the emitting plasma / below density threshold
};

}  // namespace bhsim
