#pragma once

namespace bhsim {

// -----------------------------------------------------------------------------
// Géométrie de Kerr (trou noir en rotation), coordonnées de Boyer-Lindquist.
//
//   Σ = r² + a²cos²θ ,   Δ = r² - 2Mr + a²
//   ds² = -(1 - 2Mr/Σ)dt² - (4Mar sin²θ/Σ) dt dφ + (Σ/Δ)dr² + Σ dθ²
//         + (r² + a² + 2Ma²r sin²θ/Σ) sin²θ dφ²
//
// a = J/M est le moment cinétique spécifique (longueur). a = 0 redonne
// Schwarzschild. Cette classe fournit les grandeurs géométriques, les
// quantités orbitales du disque (orbites circulaires équatoriales prograde),
// le facteur de redshift complet (gravitationnel + Doppler + frame-dragging)
// et le flux relativiste de Page-Thorne (modèle de Novikov-Thorne).
// -----------------------------------------------------------------------------
class Kerr {
public:
    Kerr(double M = 1.0, double a = 0.0) : M_(M), a_(a) {}

    double mass() const { return M_; }
    double spin() const { return a_; }              // a (longueur)
    double spinStar() const { return a_ / M_; }     // a/M ∈ [0,1)

    double delta(double r) const { return r * r - 2.0 * M_ * r + a_ * a_; }
    double sigma(double r, double cosTheta) const { return r * r + a_ * a_ * cosTheta * cosTheta; }

    // --- Tétrade ZAMO (Zero Angular Momentum Observer / observateur localement
    // non-tournant de Bardeen) à (r, θ). Repère orthonormé exact transporté par
    // l'observateur de moment cinétique nul, entraîné à ω = -g_tφ/g_φφ :
    //   e_(t) = (∂_t + ω ∂_φ)/α ,  e_(r) = √(Δ/Σ) ∂_r ,
    //   e_(θ) = ∂_θ/√Σ ,           e_(φ) = ∂_φ/√g_φφ
    // avec le lapse α = √(ΣΔ/A) et A = (r²+a²)² - a²Δ sin²θ.
    struct ZamoFrame {
        double sigma, delta, A;     // Σ, Δ, A
        double lapse, omega;        // α = √(ΣΔ/A) ,  ω = 2Mar/A
        double gphiphi;             // g_φφ = A sin²θ / Σ
        double sinTheta, cosTheta;
    };
    ZamoFrame zamo(double r, double theta) const;

    double horizonOuter() const;   // r+ = M + sqrt(M² - a²)
    double horizonInner() const;   // r- = M - sqrt(M² - a²)
    double ergosphereEquator() const { return 2.0 * M_; }
    double iscoPrograde() const;   // dernière orbite stable (formule de Bardeen)

    // Orbite circulaire équatoriale prograde (géodésique de la matière du disque).
    double orbitalOmega(double r) const;   // Ω = dφ/dt
    double orbitalUt(double r) const;      // u^t = dt/dτ

    // Facteur de redshift g = ν_obs/ν_em pour un élément du disque (orbite
    // circulaire prograde à r) vu par l'observateur ZAMO de la caméra, de lapse
    // α_obs et d'entraînement ω_obs (défauts α=1, ω=0 = observateur à l'infini).
    // Le photon a une énergie de Killing E = 1 et un moment cinétique axial Lz
    // (conservés). Inclut gravitation, Doppler, entraînement des référentiels à
    // l'émission ET à la réception :  g = (1 - ω_obs Lz) / [α_obs u^t (1 - Ω Lz)].
    double diskRedshift(double r, double Lz,
                        double alphaObs = 1.0, double omegaObs = 0.0) const;

    // Flux radiatif de Page-Thorne (forme adimensionnée, >= 0, nul au ISCO).
    // Modèle de disque mince relativiste de Novikov-Thorne.
    double pageThorneFlux(double r) const;

private:
    double M_;
    double a_;
};

}  // namespace bhsim
