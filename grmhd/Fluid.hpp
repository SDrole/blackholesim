#pragma once
#include "EOS.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// S2 — Variables fluides et conversion primitives <-> conservées.
//      HYDRODYNAMIQUE RELATIVISTE PURE : aucun champ magnétique à ce stade.
//
// ---- VARIABLES PRIMITIVES (convention HARM/iharm) ---------------------------
//   ρ      densité de masse au repos
//   u      DENSITÉ D'ÉNERGIE INTERNE (pas la pression — convention HARM)
//   ũ^i    vitesse relative à l'observateur normal
//
// Le choix de ũ^i plutôt que v^i est délibéré et détermine la robustesse de
// toute l'inversion : ũ^i n'est soumis à AUCUNE contrainte, alors que v^i
// exigerait v² < 1. Le facteur de Lorentz se reconstruit par
//        W = √(1 + γ_ij ũ^i ũ^j)
// qui est ≥ 1 par construction : une vitesse superluminique est
// STRUCTURELLEMENT impossible, elle ne peut pas être produite par une
// itération qui déraperait. En espace plat (S2), γ_ij = δ_ij et ũ^i = W v^i.
//
// ---- VARIABLES CONSERVÉES (formulation « Valencia ») ------------------------
//   D    = ρ W                  densité de masse dans le repère du labo
//   S_i  = ρ h W² v_i           densité de moment
//   τ    = ρ h W² - p - D       densité d'énergie MOINS la masse au repos
//
// Le « - D » est essentiel : sans lui, en limite non relativiste, l'énergie
// interne (petite) serait noyée dans l'énergie de masse (grande) et perdue
// par annulation catastrophique.
//
// ---- DENSITISATION : DÉLIBÉRÉMENT ABSENTE de la représentation stockée ------
// Les variables ci-dessus sont les projections eulériennes NON densitisées.
// Le facteur de volume est un multiplicateur commun à toutes les composantes :
// il SE SIMPLIFIE EXACTEMENT dans l'inversion (D, S_i, τ y apparaissent
// homogènement). L'inclure ici n'ajouterait aucune information et alourdirait
// la validation de S2. Il sera appliqué par le solveur de flux (S3/S4), où il
// a son sens géométrique, la loi de conservation s'écrivant
//        ∂_t(√γ U) + ∂_i(√γ F^i) = √γ S.
// La structure ConservedState n'aura PAS à changer pour autant.
//
// ⚠️ ATTENTION AU FACTEUR, DEUX CONVENTIONS COEXISTENT DANS LA LITTÉRATURE :
//   * Valencia (celle de nos définitions) densitise par **√γ**
//     (déterminant de la 3-métrique) :
//         D = √γ ρW ,  S_i = √γ ρhW²v_i ,  τ = √γ(ρhW² - p) - D
//   * HARM utilise les projections de COORDONNÉES T^t_ν et densitise par
//     **√-g = α√γ**.
//   Les deux coïncident pour D (√-g·ρu^t = α√γ·ρW/α = √γ·ρW), mais ce sont
//   bien deux conventions distinctes : ne pas écrire √-g pour la forme
//   Valencia.
//
// ---- CE QUI CHANGERA EN S4 / S5 ---------------------------------------------
// S4 (métrique courbe). Point important, vérifié contre la formulation 3+1
//   standard : les DÉFINITIONS de D, S_i, τ ci-dessus ne font intervenir NI
//   lapse NI shift — ce sont des projections eulériennes locales du tenseur
//   énergie-impulsion, et leurs formes algébriques restent identiques.
//   Ce qui change réellement :
//     - W = 1/√(1 - γ_ij v^i v^j)   : la 3-métrique remplace δ_ij ;
//     - les indices sont montés/baissés par γ_ij et non plus par δ_ij ;
//     - α et β^i entrent dans les FLUX (F^i = α(...) - β^i U) et dans les
//       TERMES SOURCES (dérivées de la métrique) ;
//     - α et β^i entrent aussi dans la relation entre la primitive ũ^i et la
//       quadrivitesse de coordonnées :  u^t = W/α ,  u^i = ũ^i - W β^i/α.
//   Les STRUCTURES sont inchangées ; seules les fonctions de conversion
//   deviennent métrique-dépendantes.
//   (Une version antérieure de ce commentaire affirmait que « S_i/τ font
//    intervenir lapse et shift » — c'était faux et aurait propagé une
//    convention erronée jusqu'en S4.)
// S5 (magnétisé) : ajout de B^i aux primitives et des termes de Maxwell au
//   tenseur énergie-impulsion -> extension par AJOUT de champs. L'inversion
//   magnétisée est un algorithme DIFFÉRENT (nettement plus dur, Noble et al.
//   2006), pas une retouche de celle-ci.
// -----------------------------------------------------------------------------

struct PrimitiveState {
    double rho = 0.0;        // densité de masse au repos
    double u   = 0.0;        // densité d'énergie interne
    double utilde[3] = {0.0, 0.0, 0.0};   // ũ^i (sans contrainte)
};

struct ConservedState {
    double D   = 0.0;        // ρW
    double S[3] = {0.0, 0.0, 0.0};        // ρhW²v_i
    double tau = 0.0;        // ρhW² - p - D
};

// --- Diagnostic de l'inversion ----------------------------------------------
// Exigence explicite : S3 doit pouvoir COMPTER et DIAGNOSTIQUER les échecs de
// primitive recovery, pas seulement constater qu'« ça a marché ».
enum class RecoveryFailure {
    None = 0,
    NonFiniteInput,        // NaN/Inf dans l'état conservé
    NonPositiveDensity,    // D <= 0
    EnergyBelowMomentum,   // E = τ+D <= |S| : état conservé inadmissible (v >= 1)
    BracketFailed,         // impossible d'encadrer la racine
    NoConvergence,         // itérations épuisées
    UnphysicalResult,      // solution trouvée mais hors domaine physique
    NegativeInternalEnergy // u < 0 par annulation : limite de CONDITIONNEMENT,
                            // distincte d'un échec algorithmique (voir S3 :
                            // c'est à la politique de planchers de trancher)
};

struct RecoveryStatus {
    bool converged = false;
    int  iterations = 0;
    double residual = 0.0;             // |f(p)| final
    RecoveryFailure failure = RecoveryFailure::None;

    // Diagnostics utiles au solveur (S3) pour repérer les zones limites.
    double lorentzFactor = 0.0;
    bool   usedBisection = false;      // le garde-fou a dû intervenir

    // DIAGNOSTIC NON FATAL, distinct de RecoveryFailure.
    //
    // Vrai quand l'inversion a retourné u = 0 parce que l'état conservé est
    // INDISTINGUABLE d'un fluide exactement froid à la précision disponible.
    // La récupération réussit (converged = true, failure = None) : u = 0 est
    // le meilleur état représentable. Mais l'information « était-ce un vrai
    // u = 0, ou un u > 0 tombé sous le plancher de conditionnement ? » est
    // définitivement perdue — voir la démonstration d'indistinguabilité dans
    // tests_s2.cpp (bloc D).
    //
    // Ce drapeau existe pour que S3 puisse COMPTER ces événements et décider
    // en connaissance de cause de sa politique de planchers, au lieu de les
    // voir passer silencieusement pour des fluides froids ordinaires.
    bool coldAtMachinePrecision = false;
};

const char* toString(RecoveryFailure f);

// --- Conversions -------------------------------------------------------------

// P -> U : entièrement analytique, pas d'itération.
ConservedState primitiveToConserved(const PrimitiveState& P, const IdealGasEOS& eos);

// U -> P : inversion 1D en pression, encadrée et protégée.
//
// Stratégie (exigence : ne pas faire confiance à un Newton nu) :
//   * validation préalable de l'admissibilité de l'état conservé ;
//   * encadrement explicite [p_lo, p_hi] avec vérification du changement de
//     signe AVANT d'itérer ;
//   * Newton SÉCURISÉ (« rtsafe ») : à chaque pas, si Newton sort du bracket
//     ou ne réduit pas assez l'intervalle, on bascule sur une bisection. La
//     convergence est donc garantie par la bisection, la vitesse par Newton ;
//   * AUCUN plancher correctif : un état inadmissible est SIGNALÉ, jamais
//     transformé silencieusement en état valide. Les planchers seront une
//     politique explicite du solveur en S3, pas un maquillage ici.
//
// En cas d'échec, `P` est laissé inchangé et `status.failure` en donne la
// raison.
bool conservedToPrimitive(const ConservedState& U, const IdealGasEOS& eos,
                          PrimitiveState& P, RecoveryStatus& status);

// Facteur de Lorentz associé à des primitives (espace plat : W = √(1+ũ²)).
double lorentzFactor(const PrimitiveState& P);

}  // namespace grmhd
