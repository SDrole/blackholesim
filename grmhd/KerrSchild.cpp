#include "KerrSchild.hpp"
#include "LinAlg.hpp"

#include <algorithm>
#include <cmath>

namespace grmhd {

double KerrSchild::horizonOuter() const {
    const double d = M_ * M_ - a_ * a_;
    return M_ + std::sqrt(d > 0.0 ? d : 0.0);
}

double KerrSchild::horizonInner() const {
    const double d = M_ * M_ - a_ * a_;
    return M_ - std::sqrt(d > 0.0 ? d : 0.0);
}

void KerrSchild::metricCovariant(double r, double theta, double g[4][4]) const {
    const double ct = std::cos(theta);
    // Plancher sur sinθ : l'axe polaire est une singularité de COORDONNÉES
    // (pas physique). Les codes GRMHD décalent leur grille pour ne jamais
    // évaluer exactement sur l'axe ; ce plancher est une sécurité.
    const double st = std::max(std::fabs(std::sin(theta)), 1e-14);
    const double s2 = st * st;

    const double Sigma = r * r + a_ * a_ * ct * ct;
    const double twoMrOverSigma = 2.0 * M_ * r / Sigma;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            g[i][j] = 0.0;

    g[0][0] = -(1.0 - twoMrOverSigma);
    g[0][1] = twoMrOverSigma;
    g[1][0] = g[0][1];
    g[0][3] = -twoMrOverSigma * a_ * s2;
    g[3][0] = g[0][3];

    g[1][1] = 1.0 + twoMrOverSigma;
    g[1][3] = -a_ * s2 * (1.0 + twoMrOverSigma);
    g[3][1] = g[1][3];

    g[2][2] = Sigma;

    g[3][3] = s2 * (Sigma + a_ * a_ * s2 * (1.0 + twoMrOverSigma));
}

double KerrSchild::sqrtNegGAnalytic(double r, double theta) const {
    const double ct = std::cos(theta);
    const double st = std::max(std::fabs(std::sin(theta)), 1e-14);
    const double Sigma = r * r + a_ * a_ * ct * ct;
    return Sigma * st;
}

bool KerrSchild::metricAt(double r, double theta, Metric4& out) const {
    metricCovariant(r, theta, out.g);
    if (!invert4x4(out.g, out.ginv)) return false;
    const double det = det4x4(out.g);
    if (!(det < 0.0)) return false;          // signature (-,+,+,+) => det < 0
    out.sqrtNegG = std::sqrt(-det);
    return true;
}

void KerrSchild::metricDerivatives(double r, double theta, double dg[4][4][4]) const {
    for (int l = 0; l < 4; ++l)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                dg[l][i][j] = 0.0;

    // Stationnaire (∂_t = 0) et axisymétrique (∂_φ = 0) : rien à calculer
    // pour λ = 0 et λ = 3. C'est aussi un invariant vérifiable du modèle.

    // Pas de différentiation : proportionnel à l'échelle locale, borné en
    // bas. Différences centrées d'ordre 2 => erreur O(h²) ~ 1e-11 relatif.
    const double hr = 1e-6 * std::max(std::fabs(r), 1.0);
    const double ht = 1e-6;

    double gp[4][4], gm[4][4];

    metricCovariant(r + hr, theta, gp);
    metricCovariant(r - hr, theta, gm);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dg[1][i][j] = (gp[i][j] - gm[i][j]) / (2.0 * hr);

    metricCovariant(r, theta + ht, gp);
    metricCovariant(r, theta - ht, gm);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dg[2][i][j] = (gp[i][j] - gm[i][j]) / (2.0 * ht);
}

void KerrSchild::christoffel(double r, double theta, double Gamma[4][4][4]) const {
    Metric4 m;
    if (!metricAt(r, theta, m)) {
        for (int l = 0; l < 4; ++l)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    Gamma[l][i][j] = 0.0;
        return;
    }

    double dg[4][4][4];
    metricDerivatives(r, theta, dg);

    // Γ^λ_μν = ½ g^λσ (∂_μ g_σν + ∂_ν g_σμ - ∂_σ g_μν)
    for (int l = 0; l < 4; ++l) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = mu; nu < 4; ++nu) {   // symétrique en (μ,ν)
                double sum = 0.0;
                for (int s = 0; s < 4; ++s) {
                    sum += m.ginv[l][s] * (dg[mu][s][nu] + dg[nu][s][mu] - dg[s][mu][nu]);
                }
                Gamma[l][mu][nu] = 0.5 * sum;
                Gamma[l][nu][mu] = Gamma[l][mu][nu];
            }
        }
    }
}

}  // namespace grmhd
