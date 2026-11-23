#include "Riemann.hpp"

#include <algorithm>
#include <cmath>

namespace grmhd {

void hlleFlux(const RiemannInput& in, int dir, double F[5]) {
    const SignalSpeeds sL = characteristicSpeeds(in.fL, dir);
    const SignalSpeeds sR = characteristicSpeeds(in.fR, dir);

    const double aL = std::min(0.0, std::min(sL.lambdaMinus, sR.lambdaMinus));
    const double aR = std::max(0.0, std::max(sL.lambdaPlus,  sR.lambdaPlus));

    double FL[5], FR[5];
    physicalFlux(in.fL, in.uL, dir, FL);
    physicalFlux(in.fR, in.uR, dir, FR);

    // Supersonic cases: pure upwind flux. This is required for
    // F(U,U) == F_phys(U) exactly, not merely within roundoff.
    if (aL >= 0.0) { for (int k = 0; k < 5; ++k) F[k] = FL[k]; return; }
    if (aR <= 0.0) { for (int k = 0; k < 5; ++k) F[k] = FR[k]; return; }

    const double UL[5] = {in.uL.D, in.uL.S[0], in.uL.S[1], in.uL.S[2], in.uL.tau};
    const double UR[5] = {in.uR.D, in.uR.S[0], in.uR.S[1], in.uR.S[2], in.uR.tau};

    const double inv = 1.0 / (aR - aL);
    for (int k = 0; k < 5; ++k)
        F[k] = (aR * FL[k] - aL * FR[k] + aR * aL * (UR[k] - UL[k])) * inv;
}

}  // namespace grmhd
