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
// * DÉTENTE : forme CLOSE, obtenue par intégration analytique de
//        dv/(1-v²) = sign · dp/(ρ h c_s)
//   le long de l'isentrope. Pour une Γ-law on montre (dérivation détaillée
//   dans le .cpp) que dp/(ρhc_s) = 2 dc_s/(Γ-1-c_s²), d'où
//        (1+v)/(1-v) = [(1+v_K)/(1-v_K)] · exp(2·sign·[g(c_s) - g(c_K)])
//        g(c) = (1/√(Γ-1)) · ln[(√(Γ-1)+c)/(√(Γ-1)-c)]
//   Cette forme est exacte à la précision machine : le terme « EXACT » est
//   donc justifié, sans erreur de quadrature résiduelle.
//   Une variante RK4 paramétrable est conservée UNIQUEMENT comme
//   contre-vérification indépendante (deux dérivations distinctes doivent
//   converger l'une vers l'autre — étude de convergence dans les tests).
//
// * CHOC : conditions de Rankine-Hugoniot relativistes via l'adiabatique de
//   Taub. ⚠️ FLUX DE MASSE SIGNÉ : le flux de masse propre j = ±√[(p₂-p₁)/
//   (h₁/ρ₁ - h₂/ρ₂)] porte le signe de la FAMILLE (- à gauche, + à droite).
//   N'en prendre que le module revient à appliquer la relation de saut de
//   l'onde droite aux deux familles — l'erreur est invisible sur un tube à
//   choc classique (dont l'onde gauche est une détente) mais rend FAUX tout
//   problème à deux chocs. Test de non-régression : une collision frontale
//   symétrique doit donner v* = 0 EXACTEMENT.
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

// Cause d'échec. `ok == false` doit toujours s'accompagner d'une cause : un
// oracle qui échoue sans dire pourquoi est aussi inexploitable qu'un oracle
// qui ment.
enum class RiemannFailure {
    None = 0,
    InvalidInput,      // ρ ≤ 0, p < 0, |v| ≥ 1, entrée non finie
    VacuumFormed,      // hors domaine : double détente créant du vide
    NoBracket,         // impossible d'encadrer la racine
    NotConverged       // encadrée mais non résolue à la tolérance
};

const char* toString(RiemannFailure f);

struct ExactRiemannSolution {
    bool   ok      = false;
    RiemannFailure failure = RiemannFailure::None;
    double pStar   = 0.0;   // pression au contact
    double vStar   = 0.0;   // vitesse du contact
    double rhoStarL = 0.0;  // densité à gauche du contact
    double rhoStarR = 0.0;  // densité à droite du contact
    int    iterations = 0;
    double residual = 0.0;
};

// -----------------------------------------------------------------------------
// DOMAINE DE VALIDITÉ de solveExactRiemannSRHD
// -----------------------------------------------------------------------------
// COUVERT : détente/choc à gauche × détente/choc à droite, tant qu'une région
// intermédiaire de pression STRICTEMENT POSITIVE existe.
//
// NON COUVERT : formation de VIDE. Deux états qui s'éloignent assez vite pour
// que les deux éventails de détente ne puissent plus se rejoindre créent une
// région de vide entre eux ; la structure de la solution change (il n'y a plus
// de discontinuité de contact) et l'équation v_L(p*) = v_R(p*) n'a plus de
// racine. Ce cas est DÉTECTÉ et REJETÉ (`ok=false`, `VacuumFormed`), jamais
// approximé.
//
// CRITÈRE DE VIDE (exact, dérivé de l'invariant de Riemann utilisé ci-dessus,
// donc cohérent avec la branche détente par construction). En intégrant
//   artanh(v) = artanh(v_K) + sign·[g(c_s) - g(c_K)]
// jusqu'à c_s → 0 (pression nulle), l'état gauche atteint au plus
//   v_L^vac = tanh[artanh(v_L) + g(c_L)]     (il accélère vers +x)
// et l'état droit au moins
//   v_R^vac = tanh[artanh(v_R) - g(c_R)]     (il accélère vers -x)
// Les éventails se rejoignent tant que v_L^vac > v_R^vac. Le vide apparaît donc
// ssi
//        artanh(v_R) - artanh(v_L)  ≥  g(c_L) + g(c_R)
// où g(c) = (2/√(Γ-1))·artanh(c/√(Γ-1)).
//
// CONTRÔLE INDÉPENDANT — limite newtonienne. Quand c → 0,
// g(c) = (2/√(Γ-1))·artanh(c/√(Γ-1)) → 2c/(Γ-1), et artanh(v) → v, donc le
// critère redonne  v_R - v_L ≥ 2(c_L + c_R)/(Γ-1) : exactement la condition de
// vide classique (Toro, « Riemann Solvers and Numerical Methods », §4.6).
// Cette limite est VÉRIFIÉE NUMÉRIQUEMENT dans les tests, elle n'est pas
// seulement affirmée ici.
// -----------------------------------------------------------------------------

// Vrai si (L, R) forme du vide. Exposé pour être testé indépendamment du
// solveur (et pour que la limite newtonienne soit vérifiable).
bool riemannVacuumForms(const RiemannState& L, const RiemannState& R, double gamma);

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

// --- Exposés pour l'étude de convergence de l'oracle lui-même ----------------
// Forme close (celle réellement utilisée par le solveur).
double rarefactionVelocityAnalytic(const RiemannState& pre, double pPost,
                                   double gamma, int sign);
// Quadrature RK4 à N pas — sert UNIQUEMENT à vérifier que la forme close est
// bien la limite de l'intégration numérique quand N -> ∞.
double rarefactionVelocityRK4(const RiemannState& pre, double pPost,
                              double gamma, int sign, int N);
// Compteur de déclenchements du garde-fou de vitesse dans la quadrature.
// Doit rester à 0 : aucun test ne doit passer grâce à un clamp.
long long rarefactionClampHits();
void resetRarefactionClampHits();

}  // namespace grmhd
