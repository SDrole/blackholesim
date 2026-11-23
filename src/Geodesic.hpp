#pragma once
#include <cmath>
#include "Metric.hpp"

namespace bhsim {

struct State {
    double r     = 0.0;
    double theta = 0.0;
    double phi   = 0.0;
    double pr    = 0.0;   // p_r
    double pth   = 0.0;   // p_θ
};

struct Constants {
    double E  = 1.0;
    double Lz = 0.0;
    double Q  = 0.0;
    double K  = 0.0;   // K = Q + (Lz - aE)²  (constante du tenseur de Killing)
};

inline State operator+(const State& a, const State& b) {
    return {a.r + b.r, a.theta + b.theta, a.phi + b.phi, a.pr + b.pr, a.pth + b.pth};
}
inline State operator-(const State& a, const State& b) {
    return {a.r - b.r, a.theta - b.theta, a.phi - b.phi, a.pr - b.pr, a.pth - b.pth};
}
inline State operator*(double s, const State& a) {
    return {s * a.r, s * a.theta, s * a.phi, s * a.pr, s * a.pth};
}
inline State operator*(const State& a, double s) { return s * a; }

inline double scaledErrorNorm(const State& e, const State& y, double atol, double rtol) {
    auto w = [&](double ei, double yi) {
        const double sc = atol + rtol * std::fabs(yi);
        const double q = ei / sc;
        return q * q;
    };
    const double s = w(e.r, y.r) + w(e.theta, y.theta) + w(e.phi, y.phi)
                   + w(e.pr, y.pr) + w(e.pth, y.pth);
    return std::sqrt(s / 5.0);
}

State geodesicRHS(const State& s, const Kerr& bh, const Constants& c);

double carterK(const State& s, const Kerr& bh, const Constants& c);

}  // namespace bhsim
