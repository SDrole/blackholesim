#include "Fluid.hpp"

#include <algorithm>
#include <cmath>

namespace grmhd {

const char* toString(RecoveryFailure f) {
    switch (f) {
        case RecoveryFailure::None:               return "none";
        case RecoveryFailure::NonFiniteInput:     return "non-finite input (NaN/Inf)";
        case RecoveryFailure::NonPositiveDensity: return "D <= 0";
        case RecoveryFailure::EnergyBelowMomentum:return "E <= |S| (inadmissible conserved state)";
        case RecoveryFailure::BracketFailed:      return "failed to bracket root";
        case RecoveryFailure::NoConvergence:      return "no convergence";
        case RecoveryFailure::UnphysicalResult:   return "result outside physical domain";
        case RecoveryFailure::NegativeInternalEnergy: return "u < 0 (conditioning limit)";
    }
    return "unknown";
}

double lorentzFactor(const PrimitiveState& P) {
    const double u2 = P.utilde[0]*P.utilde[0] + P.utilde[1]*P.utilde[1] + P.utilde[2]*P.utilde[2];
    return std::sqrt(1.0 + u2);
}

// =============================================================================
//  P -> U   (analytique)
// =============================================================================
ConservedState primitiveToConserved(const PrimitiveState& P, const IdealGasEOS& eos) {
    ConservedState U;
    const double W = lorentzFactor(P);
    const double p = eos.pressure(P.rho, P.u);
    const double h = eos.specificEnthalpy(P.rho, P.u);
    const double rhohW2 = P.rho * h * W * W;

    U.D = P.rho * W;
    for (int i = 0; i < 3; ++i) U.S[i] = P.rho * h * W * P.utilde[i];
    U.tau = rhohW2 - p - U.D;
    return U;
}

//
namespace {

struct Trial {
    double rho, u, W, v2, f;
    bool valid;
};

Trial evaluate(double p, double D, double S2, double tau, const IdealGasEOS& eos) {
    Trial t{};
    t.valid = false;

    const double E = tau + D;
    const double Z = E + p;
    if (!(Z > 0.0)) return t;

    const double S = std::sqrt(S2);
    if (Z <= S) return t;                       // impliquerait v >= 1

    t.v2 = S2 / (Z * Z);
    if (!(t.v2 < 1.0)) return t;

    t.W = 1.0 / std::sqrt(1.0 - t.v2);
    if (!std::isfinite(t.W) || t.W < 1.0) return t;

    t.rho = D / t.W;
    if (!(t.rho > 0.0)) return t;

    t.u = (tau - D * (t.W - 1.0)) / (t.W * t.W) - p * (1.0 - 1.0 / (t.W * t.W));

    t.f = eos.pressure(t.rho, t.u) - p;
    t.valid = std::isfinite(t.f);
    return t;
}

double dfdp(double p, double D, double S2, double tau, const IdealGasEOS& eos) {
    const double dp = 1e-8 * std::max(std::fabs(p), 1e-10);
    const Trial a = evaluate(p + dp, D, S2, tau, eos);
    const Trial b = evaluate(p - dp, D, S2, tau, eos);
    if (!a.valid || !b.valid) return 0.0;
    return (a.f - b.f) / (2.0 * dp);
}

}  // namespace

bool conservedToPrimitive(const ConservedState& U, const IdealGasEOS& eos,
                          PrimitiveState& P, RecoveryStatus& status) {
    status = RecoveryStatus{};

    //
    if (!std::isfinite(U.D) || !std::isfinite(U.tau) ||
        !std::isfinite(U.S[0]) || !std::isfinite(U.S[1]) || !std::isfinite(U.S[2])) {
        status.failure = RecoveryFailure::NonFiniteInput;
        return false;
    }
    if (!(U.D > 0.0)) {
        status.failure = RecoveryFailure::NonPositiveDensity;
        return false;
    }

    const double S2 = U.S[0]*U.S[0] + U.S[1]*U.S[1] + U.S[2]*U.S[2];
    const double S  = std::sqrt(S2);
    const double E  = U.tau + U.D;

    if (!(E > S)) {
        status.failure = RecoveryFailure::EnergyBelowMomentum;
        return false;
    }

    //
    double plo = 0.0;
    Trial tlo = evaluate(plo, U.D, S2, U.tau, eos);
    if (!tlo.valid) {
        status.failure = RecoveryFailure::BracketFailed;
        return false;
    }

    const double eps = 2.220446049250313e-16;
    const double errScale = (std::fabs(U.tau) + U.D * tlo.W) / (tlo.W * tlo.W)
                          + 0.5 * U.D * tlo.W;
    const double coldTol = 16.0 * eps * (eos.gamma() - 1.0) * errScale;

    if (tlo.f <= 0.0) {
        if (tlo.f >= -coldTol) {
            P.rho = tlo.rho;
            P.u = 0.0;
            const double Z = E + plo;
            for (int i = 0; i < 3; ++i) P.utilde[i] = tlo.W * U.S[i] / Z;
            status.converged = true;
            status.residual = std::fabs(tlo.f);
            status.lorentzFactor = tlo.W;
            status.coldAtMachinePrecision = true;
            return true;
        }
        status.failure = RecoveryFailure::UnphysicalResult;
        status.residual = std::fabs(tlo.f);
        return false;
    }

    double phi = std::max(1e-10, (eos.gamma() - 1.0) * std::max(U.tau, 1e-30));
    Trial thi = evaluate(phi, U.D, S2, U.tau, eos);
    int expand = 0;
    while ((!thi.valid || thi.f > 0.0) && expand < 200) {
        phi *= 2.0;
        thi = evaluate(phi, U.D, S2, U.tau, eos);
        ++expand;
    }
    if (!thi.valid || thi.f > 0.0) {
        status.failure = RecoveryFailure::BracketFailed;
        status.iterations = expand;
        return false;
    }

    //
    double p = 0.5 * (plo + phi);
    double dpOld = std::fabs(phi - plo);
    double dp = dpOld;
    Trial t = evaluate(p, U.D, S2, U.tau, eos);
    if (!t.valid) { p = plo; t = tlo; }

    const int maxIter = 200;
    const double tolRel = 1e-14;
    int it = 0;
    for (; it < maxIter; ++it) {
        const double d = dfdp(p, U.D, S2, U.tau, eos);

        const bool newtonOutOfRange =
            ((p - phi) * d - t.f) * ((p - plo) * d - t.f) > 0.0;
        const bool newtonTooSlow = std::fabs(2.0 * t.f) > std::fabs(dpOld * d);

        dpOld = dp;
        if (newtonOutOfRange || newtonTooSlow || d == 0.0) {
            dp = 0.5 * (phi - plo);
            p = plo + dp;
            status.usedBisection = true;
        } else {
            dp = t.f / d;
            p = p - dp;
        }

        Trial tn = evaluate(p, U.D, S2, U.tau, eos);
        if (!tn.valid) {
            dp = 0.5 * (phi - plo);
            p = plo + dp;
            tn = evaluate(p, U.D, S2, U.tau, eos);
            status.usedBisection = true;
            if (!tn.valid) {
                status.failure = RecoveryFailure::NoConvergence;
                status.iterations = it + 1;
                return false;
            }
        }
        t = tn;

        const double pScale = std::max(std::fabs(p), 1e-300);
        if (std::fabs(t.f) <= tolRel * pScale || std::fabs(dp) <= tolRel * pScale) {
            ++it;
            break;
        }

        if (t.f > 0.0) plo = p; else phi = p;
    }

    status.iterations = it;
    status.residual = std::fabs(t.f);

    if (it >= maxIter) {
        status.failure = RecoveryFailure::NoConvergence;
        return false;
    }

    //
    if (!(t.rho > 0.0) || !(t.v2 < 1.0) || !std::isfinite(t.W) || t.W < 1.0) {
        status.failure = RecoveryFailure::UnphysicalResult;
        status.lorentzFactor = t.W;
        return false;
    }
    if (!(t.u >= 0.0)) {
        status.failure = RecoveryFailure::NegativeInternalEnergy;
        status.lorentzFactor = t.W;
        return false;
    }

    P.rho = t.rho;
    P.u   = t.u;
    const double Z = E + p;
    // v^i = S^i/Z, puis ũ^i = W v^i.
    for (int i = 0; i < 3; ++i) P.utilde[i] = t.W * U.S[i] / Z;

    status.converged = true;
    status.lorentzFactor = t.W;

    if (t.u <= 16.0 * eps * (eos.gamma() - 1.0) * errScale)
        status.coldAtMachinePrecision = true;

    return true;
}

}  // namespace grmhd
