#pragma once

namespace grmhd {

// -----------------------------------------------------------------------------
// ORACLE INDÉPENDANT — solution EXACTE du problème de Riemann SRHD 1D
// (gaz parfait Γ-law, vitesses purement normales).
//
// Rôle : fournir une vérité de référence pour valider le solveur numérique,
// SANS partager ni son algèbre ni son algorithme. Le solveur numérique fait
// volumes finis + HLLE + PLM ; cet oracle fait ondes exactes + recherche de
// racine sur la pression du contact. Aucun code commun hormis l'EOS triviale.
//
// Structure de la solution (classique) : deux ondes non linéaires (choc ou
// détente) séparées par une discontinuité de contact. On cherche p* tel que
// la vitesse derrière l'onde gauche égale celle derrière l'onde droite :
//        v_L(p*) = v_R(p*)
//
// * DÉTENTE : on intègre l'invariant de Riemann différentiel
//        dv/(1-v²) = ∓ dp/(ρ h c_s)
//   le long de l'isentrope ρ(p) = ρ_K (p/p_K)^{1/Γ}, par RK4 en ln p.
//   Choix délibéré d'une INTÉGRATION NUMÉRIQUE plutôt que d'une forme close
//   mémorisée : la relation différentielle ci-dessus est élémentaire et
//   vérifiable, une formule fermée transcrite de mémoire ne le serait pas.
//
// * CHOC : conditions de Rankine-Hugoniot relativistes via l'adiabatique de
//   Taub
//        h² - h_K² = (h/ρ + h_K/ρ_K)(p - p_K)
//   résolue pour h (donc ρ), puis flux de masse propre et vitesse post-choc.
//   ⚠️ Les relations de saut sont VÉRIFIÉES NUMÉRIQUEMENT dans les tests
//   (continuité de ρWu^x, T^xx, T^{0x} à travers le choc) : si une formule
//   était mal transcrite, le test échoue. L'oracle ne s'auto-certifie pas.
// -----------------------------------------------------------------------------

struct RiemannState {
    double rho = 0.0;
    double p   = 0.0;
    double v   = 0.0;    // vitesse normale (x)
};

struct ExactRiemannSolution {
    bool   ok      = false;
    double pStar   = 0.0;   // pression au contact
    double vStar   = 0.0;   // vitesse du contact
    double rhoStarL = 0.0;  // densité à gauche du contact
    double rhoStarR = 0.0;  // densité à droite du contact
    int    iterations = 0;
    double residual = 0.0;
};

// Résout le problème de Riemann. gamma = indice adiabatique.
ExactRiemannSolution solveExactRiemannSRHD(const RiemannState& L,
                                           const RiemannState& R,
                                           double gamma);

// Échantillonne la solution auto-similaire au point xi = (x - x0)/t.
// Renvoie l'état (rho, p, v) exact en ce point.
RiemannState sampleExactRiemannSRHD(const ExactRiemannSolution& sol,
                                    const RiemannState& L, const RiemannState& R,
                                    double gamma, double xi);

// Exposé pour les tests de vérification des relations de saut :
// état post-choc et vitesse du choc, pour une onde se propageant dans l'état
// `pre` vers une pression `pPost`. `sign` = -1 pour l'onde gauche, +1 droite.
struct ShockResult {
    bool ok = false;
    double rhoPost = 0.0, vPost = 0.0, shockSpeed = 0.0;
};
ShockResult shockBranch(const RiemannState& pre, double pPost, double gamma, int sign);

}  // namespace grmhd
