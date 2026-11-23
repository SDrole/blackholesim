#include "KerrSchild.hpp"
#include "LinAlg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace grmhd {

bool KerrSchild::parametersValid() const {
    // Supported Kerr domain: M>0 and |a|<=M.
    // M<=0 is not a black hole. |a|>M is a naked singularity with no event
    // horizon. The old discriminant clamp silently mapped that invalid case to
    // an extremal horizon, so the domain is now explicit.
    if (!std::isfinite(M_) || !std::isfinite(a_)) return false;
    if (!(M_ > 0.0)) return false;
    return std::fabs(a_) <= M_;
}

bool KerrSchild::horizons(double& rPlus, double& rMinus) const {
    if (!parametersValid()) { rPlus = rMinus = 0.0; return false; }
    const double d = M_ * M_ - a_ * a_;
    // d>=0 is guaranteed by parametersValid(). max(...,0) only absorbs a
    // possible signed zero / final-roundoff artifact at exact extremality.
    const double s = std::sqrt(std::max(d, 0.0));
    rPlus  = M_ + s;
    rMinus = M_ - s;
    return true;
}

double KerrSchild::horizonOuter() const {
    double rp = 0.0, rm = 0.0;
    return horizons(rp, rm) ? rp : std::numeric_limits<double>::quiet_NaN();
}

double KerrSchild::horizonInner() const {
    double rp = 0.0, rm = 0.0;
    return horizons(rp, rm) ? rm : std::numeric_limits<double>::quiet_NaN();
}

void KerrSchild::metricCovariant(double r, double theta, double g[4][4]) const {
    const double ct = std::cos(theta);
    // sin(theta) floor: the polar axis is a coordinate singularity, not a
    // physical singularity. GRMHD grids normally avoid evaluating exactly on
    // the axis; this is a final numerical guard.
    const double st = std::max(std::fabs(std::sin(theta)), 1e-14);
    metricFromTrig<double>(r, st, ct, M_, a_, g);
}

double KerrSchild::sqrtNegGAnalytic(double r, double theta) const {
    const double ct = std::cos(theta);
    const double st = std::max(std::fabs(std::sin(theta)), 1e-14);
    const double Sigma = r * r + a_ * a_ * ct * ct;
    return Sigma * st;
}

bool KerrSchild::metricAt(double r, double theta, Metric4& out) const {
    metricCovariant(r, theta, out.g);
    if (!invert4x4(out.g, out.ginv)) return false;
    const double det = det4x4(out.g);
    if (!(det < 0.0)) return false;          // signature (-,+,+,+) => det < 0
    out.sqrtNegG = std::sqrt(-det);
    return true;
}

void KerrSchild::metricDerivatives(double r, double theta, double dg[4][4][4]) const {
    for (int l = 0; l < 4; ++l)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                dg[l][i][j] = 0.0;

    // Stationarity and axisymmetry imply no t or phi derivatives.

    // Scale-aware finite-difference steps. Second-order centered differences
    // give an O(h^2) derivative error around the 1e-11 relative scale here.
    const double hr = 1e-6 * std::max(std::fabs(r), 1.0);
    const double ht = 1e-6;

    double gp[4][4], gm[4][4];

    metricCovariant(r + hr, theta, gp);
    metricCovariant(r - hr, theta, gm);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dg[1][i][j] = (gp[i][j] - gm[i][j]) / (2.0 * hr);

    metricCovariant(r, theta + ht, gp);
    metricCovariant(r, theta - ht, gm);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dg[2][i][j] = (gp[i][j] - gm[i][j]) / (2.0 * ht);
}

bool KerrSchild::christoffel(double r, double theta, double Gamma[4][4][4]) const {
    Metric4 m;
    if (!metricAt(r, theta, m)) {
        // Zero-fill to avoid uninitialized output, but report failure
        // explicitly. Callers must never interpret this as a valid flat-space
        // connection.
        for (int l = 0; l < 4; ++l)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    Gamma[l][i][j] = 0.0;
        return false;
    }

    double dg[4][4][4];
    metricDerivatives(r, theta, dg);

    // Gamma^lambda_mu_nu =
    // 1/2 g^lambda_sigma (d_mu g_sigma_nu + d_nu g_sigma_mu - d_sigma g_mu_nu)
    for (int l = 0; l < 4; ++l) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = mu; nu < 4; ++nu) {
                double sum = 0.0;
                for (int s = 0; s < 4; ++s) {
                    sum += m.ginv[l][s] * (dg[mu][s][nu] + dg[nu][s][mu] - dg[s][mu][nu]);
                }
                Gamma[l][mu][nu] = 0.5 * sum;
                Gamma[l][nu][mu] = Gamma[l][mu][nu];
            }
        }
    }
    return true;
}

}  // namespace grmhd
