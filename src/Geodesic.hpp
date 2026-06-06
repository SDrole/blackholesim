#pragma once
#include <cmath>
#include "Metric.hpp"

namespace bhsim {

// -----------------------------------------------------------------------------
// État canonique d'un photon en coordonnées de Boyer-Lindquist.
// On intègre les équations de Hamilton issues du super-hamiltonien
//   H = ½ g^{μν} p_μ p_ν = 0   (géodésique nulle).
// t et φ étant cycliques, p_t = -E et p_φ = Lz sont conservés : on ne propage
// que (r, θ, φ, p_r, p_θ). Les points de rebroussement (p_r, p_θ changent de
// signe) sont traversés sans ambiguïté de racine carrée.
// -----------------------------------------------------------------------------
struct State {
    double r     = 0.0;
    double theta = 0.0;
    double phi   = 0.0;
    double pr    = 0.0;   // p_r
    double pth   = 0.0;   // p_θ
};

// Constantes du mouvement (calculées aux conditions initiales).
struct Constants {
    double E  = 1.0;   // énergie (normalisée à 1)
    double Lz = 0.0;   // moment cinétique axial = p_φ
    double Q  = 0.0;   // constante de Carter (nulle pour orbite équatoriale)
    double K  = 0.0;   // K = Q + (Lz - aE)²  (constante du tenseur de Killing)
};

// --- État vu comme un vecteur à 5 composantes (pour l'intégrateur) -----------
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

// Norme d'erreur pondérée (contrôle de pas Hairer-style : atol + rtol|y|).
// Indispensable ici car r ~ 1000 cohabite avec des angles ~ 1.
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

// Membre de droite des équations de Hamilton : dState/dλ.
State geodesicRHS(const State& s, const Kerr& bh, const Constants& c);

// Constante de Carter K recalculée à partir de l'état courant (pour vérifier
// sa conservation le long de l'intégration).
double carterK(const State& s, const Kerr& bh, const Constants& c);

}  // namespace bhsim
