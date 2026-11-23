#pragma once
#include <cmath>

namespace grmhd {

inline constexpr double kPi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Modified Kerr-Schild (MKS) coordinates — standard grid used by HARM-family
// codes.
//
// A raw Kerr-Schild grid does not satisfy two practical needs:
//   * resolve the region near the horizon without wasting cells far away
//     -> logarithmic radial grid: r = exp(x1)
//   * concentrate cells toward the equatorial plane, where the disk lives,
//     rather than toward the poles
//     -> theta(x2) = pi x2 + 1/2 (1-h) sin(2 pi x2),
//        x2 in [0,1], h in (0,1]
//        h = 1 gives a uniform angular grid; h -> 0 concentrates cells near
//        the equator.
//   * phi = x3 (unchanged)
//
// These mappings are analytic and invertible. Their Jacobians are used to
// transform the Kerr-Schild metric into MKS coordinates.
// -----------------------------------------------------------------------------
class ModifiedKerrSchild {
public:
    explicit ModifiedKerrSchild(double hslope = 0.3) : h_(hslope) {}

    double hslope() const { return h_; }

    double rOf(double x1) const { return std::exp(x1); }
    double x1Of(double r) const { return std::log(r); }

    double thetaOf(double x2) const {
        return kPi * x2 + 0.5 * (1.0 - h_) * std::sin(2.0 * kPi * x2);
    }

    // dr/dx1 = exp(x1) = r
    double drdx1(double x1) const { return std::exp(x1); }

    // dtheta/dx2 = pi + pi(1-h) cos(2 pi x2)
    double dthetadx2(double x2) const {
        return kPi + kPi * (1.0 - h_) * std::cos(2.0 * kPi * x2);
    }

    // Numerical inverse theta -> x2 using Newton iteration. The mapping is
    // monotone increasing for h in (0,1], so the inversion is well behaved.
    double x2Of(double theta) const {
        double x2 = theta / kPi;                 // good initial approximation
        for (int it = 0; it < 60; ++it) {
            const double f = thetaOf(x2) - theta;
            const double df = dthetadx2(x2);
            if (std::fabs(df) < 1e-300) break;
            const double dx = f / df;
            x2 -= dx;
            if (std::fabs(dx) < 1e-15) break;
        }
        return x2;
    }

    // Diagonal Jacobian dx^mu_KS / dx^nu_MKS (t and phi are unchanged).
    // Index order: 0=t, 1=r, 2=theta, 3=phi.
    void jacobianKSfromMKS(double x1, double x2, double J[4][4]) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                J[i][j] = 0.0;
        J[0][0] = 1.0;
        J[1][1] = drdx1(x1);
        J[2][2] = dthetadx2(x2);
        J[3][3] = 1.0;
    }

private:
    double h_;
};

}  // namespace grmhd
