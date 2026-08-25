#pragma once
#include "Fluid.hpp"
#include "EOS.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// S3 — Flux physiques SRHD 1D et vitesses caractéristiques.
//
// Espace de Minkowski (α=1, β^i=0, γ_ij=δ_ij), mais AUCUNE de ces formules ne
// suppose la platitude dans sa structure : elles sont écrites en fonction de
// (ρ, u, v^i, p, W), de sorte que S4 n'aura qu'à fournir γ_ij/α/β^i pour les
// généraliser sans réécrire l'interface.
//
// ---- DÉRIVATION DES FLUX (et non recopie d'un autre code) -------------------
// Tenseur énergie-impulsion d'un fluide parfait :
//     T^{μν} = ρ h u^μ u^ν + p g^{μν}
//
// Nos conservées (S2, projections eulériennes, non densitisées) :
//     D   = ρW ,   S_i = ρhW² v_i ,   τ = ρhW² - p - D
//
// Flux dans la direction x :
//   * masse   : le flux de D est ρW v^x                    -> F_D = D v^x
//   * moment  : T^x_i = ρh u^x u_i + p δ^x_i
//               = ρhW² v^x v_i + p δ^x_i                   -> F_{S_i} = S_i v^x + p δ^x_i
//   * énergie : la densité d'énergie eulérienne est E = ρhW² - p = τ + D,
//               son flux est S^x = ρhW² v^x. Le flux de τ = E - D vaut donc
//               S^x - D v^x = (ρhW² - ρW) v^x = (τ + p) v^x
//               -> F_τ = (τ + p) v^x   [les deux écritures sont identiques,
//                                        vérifié dans les tests]
//
// Vérification de cohérence avec S2 : au repos (v=0, W=1) on a D=ρ, S_i=0,
// τ=u, et TOUS les flux s'annulent sauf F_{S_x} = p. C'est exactement la
// pression qui pousse — testé explicitement.
// -----------------------------------------------------------------------------

// Variables primitives « de travail » du solveur : on manipule v^i (et non
// ũ^i) dans les flux car c'est v^i qui apparaît naturellement dans T^{μν}.
// La conversion ũ^i <-> v^i est locale et exacte : v^i = ũ^i / W.
struct FluidVars {
    double rho = 0.0;
    double u   = 0.0;      // densité d'énergie interne
    double v[3] = {0.0, 0.0, 0.0};
    double p   = 0.0;
    double W   = 1.0;
    double h   = 1.0;      // enthalpie spécifique
    double cs2 = 0.0;      // vitesse du son au carré
};

// Construit les variables de travail à partir des primitives de S2.
FluidVars makeFluidVars(const PrimitiveState& P, const IdealGasEOS& eos);

// Flux physique dans la direction `dir` (0=x, 1=y, 2=z).
// Ordre des composantes : (D, S_x, S_y, S_z, tau).
void physicalFlux(const FluidVars& f, const ConservedState& U, int dir,
                  double F[5]);

// -----------------------------------------------------------------------------
// VITESSES CARACTÉRISTIQUES RELATIVISTES
//
// ⚠️ v ± c_s est FAUX en relativité. Deux raisons distinctes :
//   1. la composition des vitesses n'est pas galiléenne ;
//   2. en présence de vitesses TRANSVERSES, les valeurs propres dépendent de
//      v² complet et pas seulement de v^x — un point qu'une simple
//      substitution « addition relativiste » manquerait.
//
// Valeurs propres du système SRHD 1D (formulation Valencia) :
//     λ_0 = v^x                                   (multiplicité 3, ondes
//                                                  matérielles / contact)
//     λ_± = [ v^x(1-c_s²) ± c_s √( (1-v²)[1 - v²c_s² - (v^x)²(1-c_s²)] ) ]
//           / (1 - v² c_s²)
// avec v² = δ_ij v^i v^j.
//
// PROVENANCE ET VÉRIFICATION : formulation standard du système SRHD en
// coordonnées eulériennes (cf. Font, Living Rev. Relativity ; Martí & Müller).
// Plutôt que de faire confiance à la transcription, les tests vérifient :
//   * v -> 0        =>  λ_± -> ± c_s
//   * c_s -> 0      =>  λ_± -> v^x  (advection pure avec le fluide)
//   * v_transverse = 0 => se réduit à l'addition relativiste
//                         (v^x ± c_s)/(1 ± v^x c_s)   [identité démontrée
//                         algébriquement dans les tests]
//   * |λ| < 1 partout (causalité) sur un large balayage
// -----------------------------------------------------------------------------
struct SignalSpeeds {
    double lambdaMinus = 0.0;
    double lambdaPlus  = 0.0;
    double lambda0     = 0.0;
};

SignalSpeeds characteristicSpeeds(const FluidVars& f, int dir);

// --- Instrumentation des GARDE-FOUS de characteristicSpeeds ------------------
// Ces clamps sont légitimes en production (ils empêchent un NaN de se propager
// depuis un état déjà hors domaine), mais en VALIDATION ils pourraient masquer
// un état non physique et faire passer un test pour la mauvaise raison.
// On les compte donc, afin de pouvoir DÉMONTRER — et non affirmer — qu'ils ne
// se déclenchent jamais dans les tests scientifiques.
struct CharacteristicClampCounters {
    long long cs2Clamped     = 0;   // c_s² ramené dans [0,1]
    long long discNegative   = 0;   // radicande < 0 -> racine forcée à 0
    long long denomDegenerate = 0;  // 1 - v²c_s² ~ 0 (exige v² = c_s² = 1)
    long long lambdaClamped  = 0;   // |λ| > 1 ramené dans [-1,1]
    long long total() const {
        return cs2Clamped + discNegative + denomDegenerate + lambdaClamped;
    }
};
const CharacteristicClampCounters& characteristicClampCounters();
void resetCharacteristicClampCounters();

}  // namespace grmhd
