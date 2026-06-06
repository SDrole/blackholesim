#pragma once
#include "Geodesic.hpp"
#include "Metric.hpp"
#include "Config.hpp"

namespace bhsim {

// Photon initial : état canonique + constantes du mouvement.
struct Photon {
    State     state;
    Constants constants;
};

// -----------------------------------------------------------------------------
// Caméra = observateur ZAMO (Zero Angular Momentum Observer, localement non-
// tournant) à distance FINIE (r_obs, θ_obs). On monte sur sa tétrade orthonormée
// exacte une caméra sténopé (focale = r_obs) visant le centre (direction radiale
// entrante -e_r) ; les axes image s'alignent sur les vecteurs locaux : β -> +e_θ
// (haut), α -> -e_φ (droite). Chaque pixel (α, β) définit une direction de visée
// unitaire n̂ dans le repère local, d'où le quadrivecteur impulsion du photon
// p^(a) = E_loc (1, n̂), transformé en coordonnées de Boyer-Lindquist par la
// tétrade. On en tire EXACTEMENT les constantes conservées (E, Lz, Q), avec la
// normalisation E = 1 (énergie de Killing) :
//   Lz = √g_φφ n_φ E_loc ,   E = (α + ω √g_φφ n_φ) E_loc ,   p_θ = √Σ n_θ E_loc ...
// À r_obs -> ∞ (α->1, ω->0), ce schéma redonne les relations de Bardeen
// (Lz = -α sinθ_obs, Q = β² + (α²-a²)cos²θ_obs) : extension exacte à distance finie.
// -----------------------------------------------------------------------------
class Camera {
public:
    Camera(const Config& cfg, const Kerr& bh);

    // Photon pour le pixel (i, j). (offX, offY) ∈ [0,1) : décalage sous-pixel.
    Photon generateRay(int i, int j, double offX = 0.5, double offY = 0.5) const;

    double observerRadius() const { return rObs_; }
    double observerLapse()  const { return obsLapse_; }   // α de la tétrade ZAMO (caméra)
    double observerOmega()  const { return obsOmega_; }   // ω (entraînement) à la caméra

private:
    Kerr   bh_;
    double rObs_, thetaObs_;
    int    width_, height_;
    double halfWidth_, halfHeight_;
    double obsLapse_ = 1.0, obsOmega_ = 0.0;   // tétrade ZAMO à (r_obs, θ_obs)
};

}  // namespace bhsim
