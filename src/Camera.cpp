#include "Camera.hpp"
#include <algorithm>
#include <cmath>

namespace bhsim {

Camera::Camera(const Config& cfg, const Kerr& bh)
    : bh_(bh), rObs_(cfg.cameraDistance), width_(cfg.width), height_(cfg.height) {
    thetaObs_ = (90.0 - cfg.cameraInclination) * kPi / 180.0;
    halfWidth_  = cfg.imageHalfWidth;
    halfHeight_ = halfWidth_ * (static_cast<double>(height_) / static_cast<double>(width_));

    const Kerr::ZamoFrame z = bh_.zamo(rObs_, thetaObs_);
    obsLapse_ = z.lapse;
    obsOmega_ = z.omega;
}

Photon Camera::generateRay(int i, int j, double offX, double offY) const {
    const double a = bh_.spin();

    const double alpha = (2.0 * (i + offX) / width_  - 1.0) * halfWidth_;
    const double beta  = (1.0 - 2.0 * (j + offY) / height_) * halfHeight_;

    const Kerr::ZamoFrame z = bh_.zamo(rObs_, thetaObs_);

    const double xh = alpha / rObs_;     // tan de l'angle image horizontal
    const double yv = beta  / rObs_;     // tan de l'angle image vertical
    double nr = -1.0, nth = yv, nph = -xh;            // β -> +e_θ , α -> -e_φ
    const double inv = 1.0 / std::sqrt(nr * nr + nth * nth + nph * nph);
    nr *= inv; nth *= inv; nph *= inv;                // |n̂| = 1

    const double sqrtGpp = std::sqrt(z.gphiphi);
    const double Etilde  = z.lapse + z.omega * sqrtGpp * nph;

    Constants c;
    c.E  = 1.0;
    c.Lz = (sqrtGpp * nph) / Etilde;

    State s;
    s.r     = rObs_;
    s.theta = thetaObs_;
    s.phi   = 0.0;
    s.pr    = std::sqrt(z.sigma / z.delta) * nr / Etilde;   // entrant (n_r < 0)
    s.pth   = std::sqrt(z.sigma)           * nth / Etilde;  // p_θ

    c.Q = s.pth * s.pth
        + z.cosTheta * z.cosTheta
          * (c.Lz * c.Lz / (z.sinTheta * z.sinTheta) - a * a * c.E * c.E);
    const double LmaE = c.Lz - a * c.E;
    c.K = c.Q + LmaE * LmaE;

    return { s, c };
}

}  // namespace bhsim
