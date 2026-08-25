#include "Flux.hpp"

#include <algorithm>
#include <cmath>

namespace grmhd {

FluidVars makeFluidVars(const PrimitiveState& P, const IdealGasEOS& eos) {
    FluidVars f;
    f.rho = P.rho;
    f.u   = P.u;
    f.W   = lorentzFactor(P);
    for (int i = 0; i < 3; ++i) f.v[i] = P.utilde[i] / f.W;   // v^i = ũ^i / W
    f.p   = eos.pressure(P.rho, P.u);
    f.h   = eos.specificEnthalpy(P.rho, P.u);
    f.cs2 = eos.soundSpeed2(P.rho, P.u);
    return f;
}

void physicalFlux(const FluidVars& f, const ConservedState& U, int dir, double F[5]) {
    const double vd = f.v[dir];
    F[0] = U.D * vd;                                    // masse
    for (int i = 0; i < 3; ++i)
        F[1 + i] = U.S[i] * vd + ((i == dir) ? f.p : 0.0);   // moment (+ p sur la diagonale)
    F[4] = (U.tau + f.p) * vd;                          // énergie (cf. dérivation en en-tête)
}

namespace {
CharacteristicClampCounters g_clamps;
}  // namespace

const CharacteristicClampCounters& characteristicClampCounters() { return g_clamps; }
void resetCharacteristicClampCounters() { g_clamps = CharacteristicClampCounters{}; }

SignalSpeeds characteristicSpeeds(const FluidVars& f, int dir) {
    SignalSpeeds s;
    const double vx = f.v[dir];
    const double v2 = f.v[0]*f.v[0] + f.v[1]*f.v[1] + f.v[2]*f.v[2];
    const double cs2 = std::clamp(f.cs2, 0.0, 1.0);
    if (cs2 != f.cs2) ++g_clamps.cs2Clamped;

    s.lambda0 = vx;

    // Cas dégénéré : fluide froid (c_s = 0) -> advection pure.
    if (cs2 <= 0.0) {
        s.lambdaMinus = vx;
        s.lambdaPlus  = vx;
        return s;
    }

    const double one_m_v2cs2 = 1.0 - v2 * cs2;
    // Radicande : (1-v²)[1 - v²c_s² - (v^x)²(1-c_s²)]
    const double inner = 1.0 - v2 * cs2 - vx * vx * (1.0 - cs2);
    const double disc  = (1.0 - v2) * inner;
    const double root  = (disc > 0.0) ? std::sqrt(disc) : 0.0;
    if (!(disc > 0.0)) ++g_clamps.discNegative;

    if (std::fabs(one_m_v2cs2) < 1e-300) {   // ne peut arriver que si v²=c_s²=1
        ++g_clamps.denomDegenerate;
        s.lambdaMinus = -1.0;
        s.lambdaPlus  =  1.0;
        return s;
    }

    const double cs = std::sqrt(cs2);
    s.lambdaMinus = (vx * (1.0 - cs2) - cs * root) / one_m_v2cs2;
    s.lambdaPlus  = (vx * (1.0 - cs2) + cs * root) / one_m_v2cs2;

    // Garde-fou de causalité. Ce n'est PAS un maquillage : les tests vérifient
    // séparément que |λ| < 1 est déjà satisfait analytiquement sur tout le
    // domaine physique. Ce clamp ne protège que contre un état d'entrée déjà
    // hors domaine (v ≥ 1), qui serait signalé ailleurs.
    const double lmC = std::clamp(s.lambdaMinus, -1.0, 1.0);
    const double lpC = std::clamp(s.lambdaPlus,  -1.0, 1.0);
    if (lmC != s.lambdaMinus) ++g_clamps.lambdaClamped;
    if (lpC != s.lambdaPlus)  ++g_clamps.lambdaClamped;
    s.lambdaMinus = lmC;
    s.lambdaPlus  = lpC;
    return s;
}

}  // namespace grmhd
