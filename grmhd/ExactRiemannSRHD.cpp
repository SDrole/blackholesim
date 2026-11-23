#include "ExactRiemannSRHD.hpp"

#include <algorithm>
#include <cmath>

namespace grmhd {

namespace {

double enthalpy(double rho, double p, double g) { return 1.0 + g / (g - 1.0) * p / rho; }
double soundSpeed(double rho, double p, double g) {
    const double h = enthalpy(rho, p, g);
    const double cs2 = g * p / (rho * h);
    return std::sqrt(std::max(cs2, 0.0));
}

//
double riemannG(double c, double g) {
    const double a = std::sqrt(g - 1.0);
    const double cc = std::min(c, a * (1.0 - 1e-15));
    return (1.0 / a) * std::log((a + cc) / (a - cc));
}

}

double rarefactionVelocityAnalytic(const RiemannState& pre, double pPost,
                                   double g, int sign) {
    if (pPost >= pre.p) return pre.v;
    const double rhoP = pre.rho * std::pow(pPost / pre.p, 1.0 / g);
    const double cK = soundSpeed(pre.rho, pre.p, g);
    const double cP = soundSpeed(rhoP, pPost, g);

    const double AK = (1.0 + pre.v) / (1.0 - pre.v);
    const double A = AK * std::exp(2.0 * static_cast<double>(sign)
                                   * (riemannG(cP, g) - riemannG(cK, g)));
    return (A - 1.0) / (A + 1.0);
}

//
long long g_rarefactionClampHits = 0;

double rarefactionVelocityRK4(const RiemannState& pre, double pPost, double g,
                              int sign, int N) {
    const double rhoK = pre.rho, pK = pre.p;
    double v = pre.v;
    if (pPost >= pK) return v;

    const double lpK = std::log(pK), lpP = std::log(std::max(pPost, 1e-300));
    const double dl = (lpP - lpK) / N;

    auto dvdlp = [&](double lp, double vv) {
        const double p = std::exp(lp);
        const double rho = rhoK * std::pow(p / pK, 1.0 / g);   // isentrope
        const double h = enthalpy(rho, p, g);
        const double cs = soundSpeed(rho, p, g);
        if (cs <= 0.0) return 0.0;
        return static_cast<double>(sign) * (1.0 - vv * vv) * p / (rho * h * cs);
    };

    double lp = lpK;
    for (int i = 0; i < N; ++i) {
        const double k1 = dvdlp(lp, v);
        const double k2 = dvdlp(lp + 0.5 * dl, v + 0.5 * dl * k1);
        const double k3 = dvdlp(lp + 0.5 * dl, v + 0.5 * dl * k2);
        const double k4 = dvdlp(lp + dl, v + dl * k3);
        v += (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        lp += dl;
        if (v < -0.999999999999 || v > 0.999999999999) ++g_rarefactionClampHits;
        v = std::clamp(v, -0.999999999999, 0.999999999999);
    }
    return v;
}

namespace {

double rarefactionVelocity(const RiemannState& pre, double pPost, double g, int sign) {
    return rarefactionVelocityAnalytic(pre, pPost, g, sign);
}

}  // namespace

//
ShockResult shockBranch(const RiemannState& pre, double pPost, double g, int sign) {
    ShockResult out;
    if (pPost <= pre.p) return out;

    const double rho1 = pre.rho, p1 = pre.p, v1 = pre.v;
    const double h1 = enthalpy(rho1, p1, g);

    auto taubResidual = [&](double h) {
        const double rho = g * pPost / ((g - 1.0) * (h - 1.0));
        return h * h - h1 * h1 - (h / rho + h1 / rho1) * (pPost - p1);
    };

    double hlo = 1.0 + 1e-14, hhi = std::max(10.0 * h1, 10.0);
    int guard = 0;
    while (taubResidual(hlo) * taubResidual(hhi) > 0.0 && guard++ < 200) hhi *= 2.0;
    if (taubResidual(hlo) * taubResidual(hhi) > 0.0) return out;

    for (int i = 0; i < 200; ++i) {
        const double hm = 0.5 * (hlo + hhi);
        if (taubResidual(hlo) * taubResidual(hm) <= 0.0) hhi = hm; else hlo = hm;
    }
    const double h2 = 0.5 * (hlo + hhi);
    const double rho2 = g * pPost / ((g - 1.0) * (h2 - 1.0));
    if (!(rho2 > 0.0)) return out;

    // Flux de masse propre :  j² = (p2 - p1)/(h1/ρ1 - h2/ρ2)
    const double denom = h1 / rho1 - h2 / rho2;
    if (!(denom > 0.0)) return out;               // choc compressif attendu
    const double j = std::sqrt((pPost - p1) / denom);

    const double W1 = 1.0 / std::sqrt(1.0 - v1 * v1);
    const double A = rho1 * rho1 * W1 * W1;
    const double disc = j * j * (j * j + A * (1.0 - v1 * v1));
    if (disc < 0.0) return out;
    const double num = A * v1 + static_cast<double>(sign) * std::sqrt(disc);
    const double Vs = num / (A + j * j);
    if (!(std::fabs(Vs) < 1.0)) return out;
    const double Ws = 1.0 / std::sqrt(1.0 - Vs * Vs);

    const double js = static_cast<double>(sign) * j;
    const double vPost = (h1 * W1 * v1 + Ws * (pPost - p1) / js)
                       / (h1 * W1 + (pPost - p1) * (Ws * v1 / js + 1.0 / (rho1 * W1)));
    if (!(std::fabs(vPost) < 1.0)) return out;

    out.ok = true;
    out.rhoPost = rho2;
    out.vPost = vPost;
    out.shockSpeed = Vs;
    return out;
}

namespace {

bool branchVelocity(const RiemannState& pre, double p, double g, int sign,
                    double& v, double& rho) {
    if (p > pre.p) {
        const ShockResult s = shockBranch(pre, p, g, sign);
        if (!s.ok) return false;
        v = s.vPost; rho = s.rhoPost;
        return true;
    }
    v = rarefactionVelocity(pre, p, g, sign);
    rho = pre.rho * std::pow(std::max(p, 1e-300) / pre.p, 1.0 / g);
    return std::fabs(v) < 1.0;
}

}  // namespace

const char* toString(RiemannFailure f) {
    switch (f) {
        case RiemannFailure::None:         return "none";
        case RiemannFailure::InvalidInput: return "invalid input (rho<=0, p<0, |v|>=1 or non-finite)";
        case RiemannFailure::VacuumFormed: return "vacuum formation (outside oracle domain)";
        case RiemannFailure::NoBracket:    return "failed to bracket root";
        case RiemannFailure::NotConverged: return "did not converge to tolerance";
    }
    return "?";
}

bool riemannVacuumForms(const RiemannState& L, const RiemannState& R, double g) {
    if (!(L.rho > 0.0) || !(R.rho > 0.0)) return false;
    if (!(std::fabs(L.v) < 1.0) || !(std::fabs(R.v) < 1.0)) return false;
    const double cL = soundSpeed(L.rho, L.p, g);
    const double cR = soundSpeed(R.rho, R.p, g);
    const double lhs = std::atanh(R.v) - std::atanh(L.v);
    const double rhs = riemannG(cL, g) + riemannG(cR, g);
    return lhs >= rhs;
}

namespace {

bool finiteState(const RiemannState& s) {
    return std::isfinite(s.rho) && std::isfinite(s.p) && std::isfinite(s.v)
        && s.rho > 0.0 && s.p >= 0.0 && std::fabs(s.v) < 1.0;
}

}  // namespace

ExactRiemannSolution solveExactRiemannSRHD(const RiemannState& L,
                                           const RiemannState& R,
                                           double g) {
    ExactRiemannSolution sol;

    //
    if (!finiteState(L) || !finiteState(R) || !(g > 1.0)) {
        sol.failure = RiemannFailure::InvalidInput;
        return sol;
    }

    //
    if (riemannVacuumForms(L, R, g)) {
        sol.failure = RiemannFailure::VacuumFormed;
        return sol;
    }

    auto residual = [&](double p, double* rl = nullptr, double* rr = nullptr) {
        double vL, vR, rhoL, rhoR;
        if (!branchVelocity(L, p, g, -1, vL, rhoL)) return 1e300;
        if (!branchVelocity(R, p, g, +1, vR, rhoR)) return 1e300;
        if (rl) *rl = rhoL;
        if (rr) *rr = rhoR;
        return vL - vR;
    };

    double plo = 1e-12 * std::min(L.p, R.p);
    double phi = 10.0 * std::max(L.p, R.p);
    double flo = residual(plo), fhi = residual(phi);
    int guard = 0;
    while (flo * fhi > 0.0 && guard++ < 200) {
        phi *= 2.0;
        fhi = residual(phi);
        if (phi > 1e30) break;
    }
    if (flo * fhi > 0.0) { sol.failure = RiemannFailure::NoBracket; return sol; }
    if (flo >= 1e299 || fhi >= 1e299) { sol.failure = RiemannFailure::NoBracket; return sol; }

    double p = 0.0;
    int it = 0;
    for (; it < 300; ++it) {
        p = 0.5 * (plo + phi);
        const double f = residual(p);
        if (std::fabs(phi - plo) <= 1e-14 * std::max(p, 1e-30)) break;
        if (flo * f <= 0.0) { phi = p; fhi = f; } else { plo = p; flo = f; }
    }

    double rhoL = 0.0, rhoR = 0.0;
    const double fFinal = residual(p, &rhoL, &rhoR);
    double vL, vR, tmp;
    branchVelocity(L, p, g, -1, vL, tmp);
    branchVelocity(R, p, g, +1, vR, tmp);

    const double vScale = std::max(1.0, std::fabs(vL) + std::fabs(vR));
    if (!std::isfinite(fFinal) || std::fabs(fFinal) > 1e-8 * vScale) {
        sol.failure = RiemannFailure::NotConverged;
        sol.residual = std::fabs(fFinal);
        sol.pStar = p;
        return sol;
    }

    sol.ok = true;
    sol.failure = RiemannFailure::None;
    sol.pStar = p;
    sol.vStar = 0.5 * (vL + vR);
    sol.rhoStarL = rhoL;
    sol.rhoStarR = rhoR;
    sol.iterations = it;
    sol.residual = std::fabs(fFinal);
    return sol;
}

RiemannState sampleExactRiemannSRHD(const ExactRiemannSolution& sol,
                                    const RiemannState& L, const RiemannState& R,
                                    double g, double xi) {
    RiemannState out;
    if (!sol.ok) return out;

    //
    if (xi <= sol.vStar) {
        if (sol.pStar > L.p) {
            const ShockResult s = shockBranch(L, sol.pStar, g, -1);
            if (s.ok && xi < s.shockSpeed) return L;
            out.rho = sol.rhoStarL; out.p = sol.pStar; out.v = sol.vStar;
            return out;
        }
        const double csL = soundSpeed(L.rho, L.p, g);
        const double headL = (L.v - csL) / (1.0 - L.v * csL);
        const double csS = soundSpeed(sol.rhoStarL, sol.pStar, g);
        const double tailL = (sol.vStar - csS) / (1.0 - sol.vStar * csS);
        if (xi <= headL) return L;
        if (xi >= tailL) { out.rho = sol.rhoStarL; out.p = sol.pStar; out.v = sol.vStar; return out; }
        double lo = sol.pStar, hi = L.p;
        for (int i = 0; i < 200; ++i) {
            const double pm = 0.5 * (lo + hi);
            const double rr = L.rho * std::pow(pm / L.p, 1.0 / g);
            const double vv = rarefactionVelocity(L, pm, g, -1);
            const double cc = soundSpeed(rr, pm, g);
            const double lam = (vv - cc) / (1.0 - vv * cc);
            if (lam < xi) hi = pm; else lo = pm;
        }
        const double pm = 0.5 * (lo + hi);
        out.p = pm;
        out.rho = L.rho * std::pow(pm / L.p, 1.0 / g);
        out.v = rarefactionVelocity(L, pm, g, -1);
        return out;
    }

    //
    if (sol.pStar > R.p) {                  // choc droit
        const ShockResult s = shockBranch(R, sol.pStar, g, +1);
        if (s.ok && xi > s.shockSpeed) return R;
        out.rho = sol.rhoStarR; out.p = sol.pStar; out.v = sol.vStar;
        return out;
    }
    const double csR = soundSpeed(R.rho, R.p, g);
    const double headR = (R.v + csR) / (1.0 + R.v * csR);
    const double csS = soundSpeed(sol.rhoStarR, sol.pStar, g);
    const double tailR = (sol.vStar + csS) / (1.0 + sol.vStar * csS);
    if (xi >= headR) return R;
    if (xi <= tailR) { out.rho = sol.rhoStarR; out.p = sol.pStar; out.v = sol.vStar; return out; }
    double lo = sol.pStar, hi = R.p;
    for (int i = 0; i < 200; ++i) {
        const double pm = 0.5 * (lo + hi);
        const double rr = R.rho * std::pow(pm / R.p, 1.0 / g);
        const double vv = rarefactionVelocity(R, pm, g, +1);
        const double cc = soundSpeed(rr, pm, g);
        const double lam = (vv + cc) / (1.0 + vv * cc);
        if (lam > xi) hi = pm; else lo = pm;
    }
    const double pm = 0.5 * (lo + hi);
    out.p = pm;
    out.rho = R.rho * std::pow(pm / R.p, 1.0 / g);
    out.v = rarefactionVelocity(R, pm, g, +1);
    return out;
}

long long rarefactionClampHits() { return g_rarefactionClampHits; }
void resetRarefactionClampHits() { g_rarefactionClampHits = 0; }

}  // namespace grmhd
