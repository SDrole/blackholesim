#pragma once

namespace grmhd {

// -----------------------------------------------------------------------------
// Métrique de Kerr en coordonnées de KERR-SCHILD sphériques (t, r, θ, φ).
//
// POURQUOI Kerr-Schild et pas Boyer-Lindquist (utilisé par le ray tracer) :
// BL est singulier à l'horizon (g_rr → ∞ quand Δ → 0), ce qui interdit de
// faire évoluer un fluide à travers l'horizon. Kerr-Schild est régulier
// partout sauf sur la singularité annulaire physique : c'est la forme
// standard des solveurs GRMHD (HARM, iharm3D, Athena++, BHAC…), qui placent
// leur frontière interne À L'INTÉRIEUR de l'horizon.
//
// Convention de signature : (-, +, +, +). Unités géométrisées G = c = 1.
// Indices : 0 = t, 1 = r, 2 = θ, 3 = φ.
//
// Σ = r² + a²cos²θ ,  Δ = r² - 2Mr + a²
//
// Composantes covariantes non nulles :
//   g_tt = -(1 - 2Mr/Σ)          g_tr = 2Mr/Σ
//   g_tφ = -2Mar sin²θ/Σ         g_rr = 1 + 2Mr/Σ
//   g_rφ = -a sin²θ (1 + 2Mr/Σ)  g_θθ = Σ
//   g_φφ = sin²θ [Σ + a² sin²θ (1 + 2Mr/Σ)]
//
// Résultat analytique exact utilisé comme test indépendant : √-g = Σ sinθ.
// -----------------------------------------------------------------------------

struct Metric4 {
    double g[4][4]    = {};   // covariant   g_μν
    double ginv[4][4] = {};   // contravariant g^μν
    double sqrtNegG   = 0.0;  // √(-det g)
};

// Cœur de la métrique, templaté sur le type scalaire et paramétré par
// (sinθ, cosθ) plutôt que par θ. Deux raisons :
//   * le plancher sur |sinθ| (singularité de coordonnées sur l'axe) reste à
//     la charge de l'appelant, ce qui garde ce cœur analytique pur ;
//   * il devient instanciable en std::complex, ce qui permet la
//     DIFFÉRENTIATION PAR PAS COMPLEXE  f'(x) ≈ Im[f(x+ih)]/h  — exacte à la
//     précision machine (aucune soustraction, donc aucune annulation
//     catastrophique), utilisée comme référence indépendante pour valider le
//     schéma de différences finies des Christoffels.
template <typename T>
void metricFromTrig(const T& r, const T& st, const T& ct,
                    const T& M, const T& a, T g[4][4]) {
    const T s2 = st * st;
    const T Sigma = r * r + a * a * ct * ct;
    const T twoMrOverSigma = T(2.0) * M * r / Sigma;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            g[i][j] = T(0.0);

    g[0][0] = -(T(1.0) - twoMrOverSigma);
    g[0][1] = twoMrOverSigma;
    g[1][0] = g[0][1];
    g[0][3] = -twoMrOverSigma * a * s2;
    g[3][0] = g[0][3];

    g[1][1] = T(1.0) + twoMrOverSigma;
    g[1][3] = -a * s2 * (T(1.0) + twoMrOverSigma);
    g[3][1] = g[1][3];

    g[2][2] = Sigma;

    g[3][3] = s2 * (Sigma + a * a * s2 * (T(1.0) + twoMrOverSigma));
}

class KerrSchild {
public:
    KerrSchild(double M = 1.0, double a = 0.0) : M_(M), a_(a) {}

    double mass() const { return M_; }
    double spin() const { return a_; }

    // Rayons des horizons : r± = M ± √(M² - a²).
    double horizonOuter() const;
    double horizonInner() const;

    // Métrique covariante seule (analytique).
    void metricCovariant(double r, double theta, double g[4][4]) const;

    // Métrique complète : covariante + inverse (numérique) + √-g.
    // L'inverse est calculé par Gauss-Jordan plutôt que par une formule
    // analytique : moins de risque d'erreur de transcription, et le test
    // g·g⁻¹ = I valide alors réellement l'inverseur.
    bool metricAt(double r, double theta, Metric4& out) const;

    // √-g par la formule analytique Σ sinθ (référence de validation).
    double sqrtNegGAnalytic(double r, double theta) const;

    // Symboles de Christoffel Γ^λ_μν par différentiation numérique de la
    // métrique — approche standard des codes HARM : plus robuste que de
    // transcrire à la main les expressions analytiques (longues et
    // facilement fausses), et l'erreur de troncature reste très inférieure
    // aux erreurs du schéma hydrodynamique.
    //   Γ^λ_μν = ½ g^λσ (∂_μ g_σν + ∂_ν g_σμ - ∂_σ g_μν)
    // Kerr en KS est stationnaire et axisymétrique : ∂_t g = ∂_φ g = 0,
    // seules les dérivées ∂_r et ∂_θ sont non nulles.
    void christoffel(double r, double theta, double Gamma[4][4][4]) const;

    // Dérivées ∂_λ g_μν (numériques). Exposé pour le test de compatibilité
    // métrique ∇_λ g_μν = 0.
    void metricDerivatives(double r, double theta, double dg[4][4][4]) const;

private:
    double M_;
    double a_;
};

}  // namespace grmhd
