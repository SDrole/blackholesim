#pragma once
#include "Flux.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// Solveur de Riemann approché — interface conçue pour être remplaçable.
//
// Choix de baseline : HLLE. Décision assumée de préférer un HLLE bien validé
// à un HLLC introduit trop tôt :
//   * robuste sur les chocs forts, très peu de branches pathologiques ;
//   * se généralise naturellement vers GRHD puis GRMHD ;
//   * excellent point de référence pour distinguer un BUG PHYSIQUE d'un simple
//     manque de résolution — un HLLC qui « marche mieux » masquerait cette
//     distinction au moment où l'on en a le plus besoin.
// Contrepartie connue et assumée : HLLE est diffusif sur les discontinuités
// de contact (il ne restitue pas l'onde intermédiaire). Cela se voit sur les
// tubes à choc et n'est PAS un bug — c'est mesuré dans les tests.
//
// L'interface prend les états gauche/droite RECONSTRUITS et rend le flux de
// face. Ajouter HLLC/HLLD plus tard = ajouter une fonction de même signature,
// sans toucher au reste du solveur.
// -----------------------------------------------------------------------------

struct RiemannInput {
    FluidVars      fL, fR;    // variables de travail de part et d'autre
    ConservedState uL, uR;    // conservées correspondantes
};

// Flux numérique HLLE dans la direction `dir`.
//   a_L = min(0, λ⁻_L, λ⁻_R) ,  a_R = max(0, λ⁺_L, λ⁺_R)
//   F = (a_R F_L - a_L F_R + a_R a_L (U_R - U_L)) / (a_R - a_L)
// Les cas supersoniques (a_L ≥ 0 ou a_R ≤ 0) rendent directement le flux amont,
// ce qui garantit F(U,U) = F_phys(U) exactement (testé).
void hlleFlux(const RiemannInput& in, int dir, double F[5]);

}  // namespace grmhd
