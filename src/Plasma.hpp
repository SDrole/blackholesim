#pragma once

namespace bhsim {

// -----------------------------------------------------------------------------
// État local du plasma en un point de l'espace-temps — interface commune entre
// le backend procédural actuel (texture MHD stylisée, cf. Renderer.cpp) et un
// futur backend GRMHD (ingestion de snapshots réels au format iharm HDF5).
// Voir docs/SESSION_LOG_AND_ROADMAP.md pour le plan complet (Track A/B).
//
// Toutes les grandeurs sont en unités géométrisées (G = c = 1), cohérentes
// avec le reste du moteur. Un backend GRMHD futur peuplera ces mêmes champs
// par interpolation dans un cube de données au lieu de les calculer
// analytiquement — le reste du pipeline (synchrotron, transfert polarisé) ne
// dépend que de cette structure, jamais de la source des données.
// -----------------------------------------------------------------------------
struct PlasmaState {
    double rho     = 0.0;   // densité (ici : facteur d'opacité relatif, sans dimension)
    double Theta_e = 0.0;   // température électronique adimensionnée kT_e/(m_e c²)
                             // — non peuplée avant l'étape 2 (prescription R-high/R-low)

    // Quadrivitesse du fluide (contravariant, coordonnées de Boyer-Lindquist
    // locales) : u^t, u^r, u^theta, u^phi.
    double u4[4] = {0.0, 0.0, 0.0, 0.0};

    // Quadrivecteur champ magnétique (contravariant, même repère). Dans le
    // backend procédural actuel, c'est un champ JOUET (toroïdal, aligné sur
    // l'écoulement, magnitude arbitraire) — pas une vraie sortie GRMHD. Non
    // utilisé avant les étapes 3-4 (coefficients synchrotron / polarisation).
    double B4[4] = {0.0, 0.0, 0.0, 0.0};

    bool valid = false;   // hors du tore de plasma / densité sous le plancher
};

}  // namespace bhsim
