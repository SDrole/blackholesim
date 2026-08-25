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

// Contenu relu — COUVRE TOUS les datasets et attributs réellement écrits.
// Un aller-retour qui ne relit qu'une partie des champs ne teste pas le
// writer : il teste les champs qu'on a pensé à relire.
struct SnapshotContents {
    std::vector<double> x, rho, uu, u1, u2, u3;
    double t = 0.0;
    double gam = 0.0;
    int nx = 0;
    std::string stage;          // bhsim_stage
    std::string metric;         // metric
    std::string absentFields;   // absent_fields
};

bool readSnapshotHDF5(const std::string& path, SnapshotContents& out);

}  // namespace grmhd
