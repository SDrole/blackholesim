#include "Camera.hpp"
#include <algorithm>
#include <cmath>

namespace bhsim {

Camera::Camera(const Config& cfg, const Kerr& bh)
    : bh_(bh), rObs_(cfg.cameraDistance), width_(cfg.width), height_(cfg.height) {
    // Inclinaison donnée au-dessus du plan équatorial -> angle polaire θ_obs.
    thetaObs_ = (90.0 - cfg.cameraInclination) * kPi / 180.0;
    halfWidth_  = cfg.imageHalfWidth;
    halfHeight_ = halfWidth_ * (static_cast<double>(height_) / static_cast<double>(width_));

    // Tétrade ZAMO de la caméra : lapse α et entraînement ω à (r_obs, θ_obs).
    const Kerr::ZamoFrame z = bh_.zamo(rObs_, thetaObs_);
    obsLapse_ = z.lapse;
    obsOmega_ = z.omega;
}

Photon Camera::generateRay(int i, int j, double offX, double offY) const {
    const double a = bh_.spin();

    // Plan image : α horizontal (droite +), β vertical (haut +), en M.
    const double alpha = (2.0 * (i + offX) / width_  - 1.0) * halfWidth_;
    const double beta  = (1.0 - 2.0 * (j + offY) / height_) * halfHeight_;

    // Tétrade ZAMO exacte au point de la caméra (r_obs, θ_obs).
    const Kerr::ZamoFrame z = bh_.zamo(rObs_, thetaObs_);

    // Caméra sténopé (focale = r_obs) montée sur le ZAMO, visant le centre :
    // direction radiale entrante -e_r, axes image alignés sur (e_θ, e_φ).
    // Direction de visée unitaire n̂ dans le repère orthonormé (e_r, e_θ, e_φ).
    const double xh = alpha / rObs_;     // tan de l'angle image horizontal
    const double yv = beta  / rObs_;     // tan de l'angle image vertical
    double nr = -1.0, nth = yv, nph = -xh;            // β -> +e_θ , α -> -e_φ
    const double inv = 1.0 / std::sqrt(nr * nr + nth * nth + nph * nph);
    nr *= inv; nth *= inv; nph *= inv;                // |n̂| = 1

    // Quadrivecteur impulsion local du photon : p^(a) = E_loc (1, n̂). Passage
    // repère ZAMO -> coordonnées : Lz = √g_φφ n_φ E_loc, E = (α + ω√g_φφ n_φ)E_loc,
    // p_θ = √Σ n_θ E_loc, p_r = √(Σ/Δ) n_r E_loc. On normalise E = 1 (énergie de
    // Killing) en posant E_loc = 1/Ẽ, ce qui rescale tout le covecteur p_μ par
    // 1/Ẽ (la géodésique nulle est inchangée : H = 0 préservé).
    const double sqrtGpp = std::sqrt(z.gphiphi);
    const double Etilde  = z.lapse + z.omega * sqrtGpp * nph;     // E pour E_loc = 1

    Constants c;
    c.E  = 1.0;
    c.Lz = (sqrtGpp * nph) / Etilde;

    State s;
    s.r     = rObs_;
    s.theta = thetaObs_;
    s.phi   = 0.0;
    s.pr    = std::sqrt(z.sigma / z.delta) * nr / Etilde;   // entrant (n_r < 0)
    s.pth   = std::sqrt(z.sigma)           * nth / Etilde;  // p_θ

    // Constante de Carter Q (photon, μ = 0) et K du tenseur de Killing, cohérentes
    // avec carterK() : Q = p_θ² + cos²θ ( Lz²/sin²θ - a²E² ).
    c.Q = s.pth * s.pth
        + z.cosTheta * z.cosTheta
          * (c.Lz * c.Lz / (z.sinTheta * z.sinTheta) - a * a * c.E * c.E);
    const double LmaE = c.Lz - a * c.E;
    c.K = c.Q + LmaE * LmaE;

    return { s, c };
}

}  // namespace bhsim
