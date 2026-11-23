#include "Geodesic.hpp"
#include <algorithm>
#include <cmath>

namespace bhsim {

State geodesicRHS(const State& s, const Kerr& bh, const Constants& c) {
    const double M = bh.mass();
    const double a = bh.spin();
    const double E = c.E, Lz = c.Lz;

    const double r = s.r;
    const double sinT = std::max(std::sin(s.theta), 1e-6);
    const double cosT = std::cos(s.theta);
    const double s2 = sinT * sinT;

    const double Sigma = r * r + a * a * cosT * cosT;
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double Dsafe = (std::fabs(Delta) < 1e-12) ? std::copysign(1e-12, Delta) : Delta;

    const double P = E * (r * r + a * a) - a * Lz;

    const double G = Delta * s.pr * s.pr + s.pth * s.pth - P * P / Dsafe
                   + a * a * E * E * s2 - 2.0 * a * E * Lz + Lz * Lz / s2;

    const double dDelta = 2.0 * r - 2.0 * M;          // ∂Δ/∂r
    const double dP     = 2.0 * r * E;                // ∂P/∂r
    const double dG_dr  = dDelta * s.pr * s.pr
                        - 2.0 * P * dP / Dsafe
                        + P * P * dDelta / (Dsafe * Dsafe);
    const double dG_dth = 2.0 * a * a * E * E * sinT * cosT
                        - 2.0 * Lz * Lz * cosT / (sinT * s2);   // -2Lz²cosθ/sin³θ

    const double dSig_dr  = 2.0 * r;
    const double dSig_dth = -2.0 * a * a * cosT * sinT;
    const double invSig   = 1.0 / Sigma;
    const double invSig2  = invSig * invSig;

    State d;
    // Positions :  dx^i/dλ = ∂H/∂p_i.
    d.r   = (Delta * invSig) * s.pr;
    d.theta = invSig * s.pth;
    d.phi = invSig * (a * P / Dsafe - a * E + Lz / s2);
    // Moments :  dp_i/dλ = -∂H/∂x^i = -∂G/∂x^i/(2Σ) + G ∂Σ/∂x^i/(2Σ²).
    d.pr  = -0.5 * dG_dr * invSig + 0.5 * G * dSig_dr * invSig2;
    d.pth = -0.5 * dG_dth * invSig + 0.5 * G * dSig_dth * invSig2;
    return d;
}

double carterK(const State& s, const Kerr& bh, const Constants& c) {
    const double a = bh.spin();
    const double E = c.E, Lz = c.Lz;
    const double sinT = std::max(std::sin(s.theta), 1e-6);
    const double cosT = std::cos(s.theta);

    // Q = p_θ² + cos²θ ( Lz²/sin²θ - a²E² ) ,  puis  K = Q + (Lz - aE)².
    const double Q = s.pth * s.pth + cosT * cosT * (Lz * Lz / (sinT * sinT) - a * a * E * E);
    const double LmaE = Lz - a * E;
    return Q + LmaE * LmaE;
}

}  // namespace bhsim
