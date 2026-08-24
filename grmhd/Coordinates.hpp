#pragma once
#include <cmath>

namespace grmhd {

inline constexpr double kPi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Coordonnées MKS (Modified Kerr-Schild) — maillage standard des codes HARM.
//
// Deux besoins que le maillage KS brut ne satisfait pas :
//   * résoudre finement près de l'horizon sans gaspiller de cellules au loin
//     -> maillage logarithmique en rayon :  r = exp(x¹)
//   * concentrer les cellules vers le plan équatorial (là où se trouve le
//     disque) plutôt que vers les pôles
//     -> θ(x²) = π x² + ½(1 - h) sin(2π x²),  x² ∈ [0,1], h ∈ (0,1]
//        h = 1 redonne un maillage uniforme ; h → 0 concentre à l'équateur.
//   * φ = x³ (inchangé)
//
// Ces transformations sont analytiques et inversibles ; leurs jacobiens
// servent à transporter la métrique de KS vers MKS.
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

    // dr/dx¹ = exp(x¹) = r
    double drdx1(double x1) const { return std::exp(x1); }

    // dθ/dx² = π + π(1 - h) cos(2π x²)
    double dthetadx2(double x2) const {
        return kPi + kPi * (1.0 - h_) * std::cos(2.0 * kPi * x2);
    }

    // Inversion numérique θ -> x² (Newton ; la fonction est monotone
    // croissante pour h ∈ (0,1], donc convergence garantie).
    double x2Of(double theta) const {
        double x2 = theta / kPi;                 // bonne approximation initiale
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

    // Jacobien diagonal dx^μ_KS / dx^ν_MKS (t et φ sont identiques).
    // Ordre des indices : 0 = t, 1 = r, 2 = θ, 3 = φ.
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
