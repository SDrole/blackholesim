#pragma once
#include <cmath>

namespace grmhd {

// -----------------------------------------------------------------------------
// Équation d'état — ISOLÉE derrière une interface minimale.
//
// Motif : ne pas disperser Γ et les relations thermodynamiques dans le
// solveur. Tout ce qui suppose « gaz parfait » doit passer par ici, pour
// qu'un changement d'EOS (Mignone & McKinney / Synge, en S5+ si le besoin
// apparaît) reste local.
//
// Convention HARM : la variable thermodynamique primitive est la DENSITÉ
// D'ÉNERGIE INTERNE `u` (et non la pression). ε = u/ρ est l'énergie interne
// spécifique, h = 1 + ε + p/ρ l'enthalpie spécifique.
//
// Γ-law : p = (Γ-1) u.
// -----------------------------------------------------------------------------
class IdealGasEOS {
public:
    explicit IdealGasEOS(double gamma = 4.0 / 3.0) : gamma_(gamma) {}

    double gamma() const { return gamma_; }

    // p(ρ, u) = (Γ-1) u   — indépendante de ρ pour une Γ-law.
    double pressure(double /*rho*/, double u) const { return (gamma_ - 1.0) * u; }

    // u(ρ, p) : inverse de la précédente.
    double internalEnergy(double /*rho*/, double p) const { return p / (gamma_ - 1.0); }

    // Enthalpie spécifique h = 1 + ε + p/ρ = 1 + Γ u / ρ  (pour une Γ-law).
    double specificEnthalpy(double rho, double u) const {
        return 1.0 + gamma_ * u / rho;
    }

    // Vitesse du son au carré, forme relativiste : c_s² = Γ p / (ρ h).
    // Non utilisée en S2 (pas de flux), mais définie ici pour que S3 la
    // trouve au bon endroit plutôt que de la réinventer dans le solveur.
    double soundSpeed2(double rho, double u) const {
        const double p = pressure(rho, u);
        const double h = specificEnthalpy(rho, u);
        return (h > 0.0) ? (gamma_ * p / (rho * h)) : 0.0;
    }

private:
    double gamma_;
};

}  // namespace grmhd
