#pragma once
#include <string>
#include "Solver1D.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// Writer HDF5 minimal — dette de la roadmap arrivant à échéance en S3.
//
// Objectif à ce stade : que l'INFRASTRUCTURE de sérialisation fonctionne et
// soit testée, pas produire un snapshot GRMHD complet (qui n'aurait aucun
// sens : il n'y a ni champ magnétique ni métrique courbe avant S5/S4).
//
// ---- ORIENTATION VERS LE CONTRAT iharm/HARM ---------------------------------
// Les noms de datasets suivent la convention iharm (`prims/RHO`, `prims/UU`,
// `prims/U1..U3`) afin que le lecteur de S4 n'ait pas à gérer deux schémas.
//
// ⚠️ CHAMPS DÉLIBÉRÉMENT ABSENTS À CE STADE — documentés comme absents plutôt
// qu'inventés avec des valeurs factices (règle : ne pas fabriquer de données) :
//   * `prims/B1..B3`  : pas de champ magnétique avant S5.
//   * métrique / grille FMKS, spin `a`, lapse, shift : espace de Minkowski ici,
//     il n'y a pas de métrique courbe à écrire.
//   * `t` est écrit, mais c'est un temps de Minkowski, pas un temps de
//     coordonnée de Kerr-Schild.
// L'attribut `bhsim_stage` = "S3-SRHD-flat" marque explicitement le niveau de
// complétude du fichier, pour qu'un lecteur ne le confonde jamais avec un vrai
// snapshot GRMHD.
// -----------------------------------------------------------------------------

// Écrit l'état courant du solveur. Renvoie false en cas d'échec HDF5.
bool writeSnapshotHDF5(const Solver1D& solver, const std::string& path);

// Relecture minimale (utilisée par les tests : un writer non relu n'est pas
// un writer testé). Remplit les vecteurs avec les données du fichier.
bool readSnapshotHDF5(const std::string& path,
                      std::vector<double>& x,
                      std::vector<double>& rho,
                      std::vector<double>& uu,
                      std::vector<double>& u1,
                      double& t, int& nx);

}  // namespace grmhd
