#pragma once
#include <vector>
#include <cstddef>
#include "Coordinates.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// S0 — Grille MKS pour le solveur GRMHD.
//
// Convention volumes finis : les variables vivent au CENTRE des cellules,
// les flux sur les FACES. On stocke les deux jeux de coordonnées.
//
// Choix délibéré (standard du domaine) : la frontière radiale interne est
// placée À L'INTÉRIEUR de l'horizon. Comme aucune information ne peut en
// sortir (causalité), la condition aux limites interne n'influence pas la
// solution physique — c'est précisément ce que permet Kerr-Schild et
// qu'interdit Boyer-Lindquist.
//
// Halo de cellules fantômes (ghost) pour la reconstruction : NG=3 couvre
// PLM (besoin de 1-2) et laisse la place à PPM/WENO plus tard sans
// retoucher la structure.
// -----------------------------------------------------------------------------
struct Grid {
    // Nombre de cellules physiques (hors ghosts) par direction.
    int n1 = 0, n2 = 0, n3 = 0;
    static constexpr int NG = 3;          // cellules fantômes de chaque côté

    // Bornes du domaine en coordonnées MKS.
    double x1min = 0.0, x1max = 0.0;      // ln r
    double x2min = 0.0, x2max = 1.0;      // θ paramétré
    double x3min = 0.0, x3max = 2.0 * kPi;

    double dx1 = 0.0, dx2 = 0.0, dx3 = 0.0;

    ModifiedKerrSchild coords{0.3};

    // Tailles allouées (avec ghosts).
    int t1() const { return n1 + 2 * NG; }
    int t2() const { return n2 + 2 * NG; }
    int t3() const { return n3 + 2 * NG; }
    std::size_t cellCount() const {
        return static_cast<std::size_t>(t1()) * t2() * t3();
    }

    // Indexation linéaire (i,j,k en indices AVEC ghosts, 0-based).
    std::size_t idx(int i, int j, int k) const {
        return (static_cast<std::size_t>(i) * t2() + j) * t3() + k;
    }

    // Coordonnées MKS du centre de la cellule (indices avec ghosts).
    double x1c(int i) const { return x1min + (i - NG + 0.5) * dx1; }
    double x2c(int j) const { return x2min + (j - NG + 0.5) * dx2; }
    double x3c(int k) const { return x3min + (k - NG + 0.5) * dx3; }

    // Coordonnées MKS de la face basse de la cellule.
    double x1f(int i) const { return x1min + (i - NG) * dx1; }
    double x2f(int j) const { return x2min + (j - NG) * dx2; }
    double x3f(int k) const { return x3min + (k - NG) * dx3; }

    // Coordonnées physiques Kerr-Schild correspondantes.
    double rOf(int i) const { return coords.rOf(x1c(i)); }
    double thetaOf(int j) const { return coords.thetaOf(x2c(j)); }
    double phiOf(int k) const { return x3c(k); }

    // Construit une grille sphérique MKS. rIn doit être placé à l'intérieur
    // de l'horizon (cf. commentaire d'en-tête).
    static Grid makeSpherical(int n1, int n2, int n3,
                              double rIn, double rOut, double hslope = 0.3) {
        Grid g;
        g.n1 = n1; g.n2 = n2; g.n3 = n3;
        g.coords = ModifiedKerrSchild(hslope);
        g.x1min = std::log(rIn);
        g.x1max = std::log(rOut);
        g.x2min = 0.0; g.x2max = 1.0;
        g.x3min = 0.0; g.x3max = 2.0 * kPi;
        g.dx1 = (g.x1max - g.x1min) / n1;
        g.dx2 = (g.x2max - g.x2min) / n2;
        g.dx3 = (g.x3max - g.x3min) / n3;
        return g;
    }
};

}  // namespace grmhd
