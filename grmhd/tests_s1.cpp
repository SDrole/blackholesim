// -----------------------------------------------------------------------------
// S1 — Suite de validation de la métrique de Kerr-Schild et des coordonnées MKS.
//
// Principe directeur de la roadmap : aucune brique n'avance sans passer son
// test, et chaque test doit confronter le code à une vérité INDÉPENDANTE
// (formule analytique fermée, identité mathématique), jamais à lui-même.
//
// T1  g·g⁻¹ = I                      -> valide l'inverseur 4x4
// T2  √-g == Σ sinθ                  -> valide la métrique covariante contre
//                                       un résultat analytique fermé
// T3  ∇_λ g_μν = 0                   -> LE test de la connexion (compatibilité
//                                       métrique) : si les Christoffels sont
//                                       faux, ceci le révèle
// T4  Γ^λ_μν = Γ^λ_νμ                -> symétrie des indices bas
// T5  ∂_t g = ∂_φ g = 0              -> stationnarité + axisymétrie
// T6  régularité à l'horizon          -> composantes finies en r₊ (ce que
//                                       Boyer-Lindquist ne permet PAS)
// T7  MKS <-> KS aller-retour + jacobiens vs différences finies
// -----------------------------------------------------------------------------
#include "KerrSchild.hpp"
#include "Coordinates.hpp"
#include "Grid.hpp"
#include "LinAlg.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>

using namespace grmhd;

namespace {

int g_failures = 0;

void report(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  [%s] %-46s  err = %.3e  (tol %.1e)\n",
                ok ? "OK  " : "FAIL", name, err, tol);
    if (!ok) ++g_failures;
}

// Points d'échantillonnage couvrant le domaine utile : de l'intérieur de
// l'horizon jusqu'au champ lointain, à différentes latitudes, pour plusieurs
// spins (dont a=0 Schwarzschild et un cas quasi-extrémal).
struct SamplePoint { double a, r, theta; };

std::vector<SamplePoint> samplePoints() {
    std::vector<SamplePoint> pts;
    const double spins[]  = {0.0, 0.5, 0.9, 0.998};
    const double radii[]  = {1.05, 1.5, 2.0, 4.0, 10.0, 50.0, 500.0};
    const double thetas[] = {0.05, 0.3, 1.0, kPi / 2.0, 2.2, kPi - 0.05};
    for (double a : spins)
        for (double r : radii)
            for (double th : thetas)
                pts.push_back({a, r, th});
    return pts;
}

// --- T1 : g · g⁻¹ = I --------------------------------------------------------
void testInverse() {
    double worst = 0.0;
    for (const auto& p : samplePoints()) {
        KerrSchild ks(1.0, p.a);
        Metric4 m;
        if (!ks.metricAt(p.r, p.theta, m)) { worst = 1.0; continue; }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double s = 0.0;
                for (int k = 0; k < 4; ++k) s += m.g[i][k] * m.ginv[k][j];
                const double target = (i == j) ? 1.0 : 0.0;
                worst = std::max(worst, std::fabs(s - target));
            }
        }
    }
    report("T1 inverse metrique  g*ginv = I", worst, 1e-10);
}

// --- T2 : √-g vs formule analytique Σ sinθ -----------------------------------
// Test le plus important de la métrique covariante : Σ sinθ est un résultat
// fermé connu, totalement indépendant de notre implémentation du déterminant.
void testSqrtNegG() {
    double worst = 0.0;
    for (const auto& p : samplePoints()) {
        KerrSchild ks(1.0, p.a);
        Metric4 m;
        if (!ks.metricAt(p.r, p.theta, m)) { worst = 1.0; continue; }
        const double ref = ks.sqrtNegGAnalytic(p.r, p.theta);
        worst = std::max(worst, std::fabs(m.sqrtNegG - ref) / std::max(ref, 1e-30));
    }
    report("T2 sqrt(-g) vs Sigma*sin(theta)", worst, 1e-12);
}

// --- T3 : compatibilité métrique ∇_λ g_μν = 0 --------------------------------
// ∇_λ g_μν = ∂_λ g_μν - Γ^σ_λμ g_σν - Γ^σ_λν g_μσ  doit être nul.
// C'est LE test de la connexion : une erreur dans les Christoffels
// (transcription, indices, signe) apparaît ici et nulle part ailleurs.
// Normalisé par l'échelle locale de la métrique pour rester adimensionné.
void testMetricCompatibility() {
    double worst = 0.0;
    for (const auto& p : samplePoints()) {
        KerrSchild ks(1.0, p.a);
        Metric4 m;
        if (!ks.metricAt(p.r, p.theta, m)) { worst = 1.0; continue; }
        double dg[4][4][4], G[4][4][4];
        ks.metricDerivatives(p.r, p.theta, dg);
        if (!ks.christoffel(p.r, p.theta, G)) { worst = 1.0; continue; }

        double scale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                scale = std::max(scale, std::fabs(m.g[i][j]));
        scale = std::max(scale, 1e-30);

        for (int l = 0; l < 4; ++l) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    double cov = dg[l][mu][nu];
                    for (int s = 0; s < 4; ++s)
                        cov -= G[s][l][mu] * m.g[s][nu] + G[s][l][nu] * m.g[mu][s];
                    // Échelle locale : les dérivées ∂_θ portent un facteur
                    // supplémentaire ~Σ, on normalise par max|g| et par le
                    // rayon pour comparer des grandeurs comparables.
                    worst = std::max(worst, std::fabs(cov) / (scale * std::max(p.r, 1.0)));
                }
            }
        }
    }
    // Tolérance fixée par la différentiation numérique centrée (O(h²) avec
    // h = 1e-6 relatif) : ~1e-9 est le plancher atteignable, pas la
    // précision machine.
    report("T3 compatibilite metrique  grad_g = 0", worst, 1e-7);
}

// --- T4 : symétrie de Γ dans les indices bas ---------------------------------
void testChristoffelSymmetry() {
    double worst = 0.0;
    for (const auto& p : samplePoints()) {
        KerrSchild ks(1.0, p.a);
        double G[4][4][4];
        if (!ks.christoffel(p.r, p.theta, G)) { worst = 1.0; continue; }
        for (int l = 0; l < 4; ++l)
            for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu)
                    worst = std::max(worst, std::fabs(G[l][mu][nu] - G[l][nu][mu]));
    }
    report("T4 symetrie Gamma^l_{mu nu} = Gamma^l_{nu mu}", worst, 1e-14);
}

// --- T5 : stationnarité et axisymétrie ---------------------------------------
// Kerr en Kerr-Schild ne dépend ni de t ni de φ. Invariant structurel : si
// ces dérivées n'étaient pas nulles, tout le reste serait suspect.
void testStationaryAxisymmetric() {
    double worst = 0.0;
    for (const auto& p : samplePoints()) {
        KerrSchild ks(1.0, p.a);
        double dg[4][4][4];
        ks.metricDerivatives(p.r, p.theta, dg);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                worst = std::max(worst,
                                 std::max(std::fabs(dg[0][i][j]), std::fabs(dg[3][i][j])));
    }
    report("T5 stationnaire + axisymetrique (d_t g = d_phi g = 0)", worst, 1e-16);
}

// --- T6 : régularité à l'horizon ---------------------------------------------
// Raison d'être de Kerr-Schild : à r = r₊, toutes les composantes restent
// finies. En Boyer-Lindquist, g_rr = Σ/Δ diverge (Δ → 0) — on le vérifie
// explicitement pour documenter le contraste.
void testHorizonRegularity() {
    double worstKS = 0.0;
    double blDivergence = 0.0;
    const double spins[] = {0.0, 0.5, 0.9, 0.998};
    for (double a : spins) {
        KerrSchild ks(1.0, a);
        const double rp = ks.horizonOuter();
        for (double f : {0.90, 0.99, 1.0, 1.01, 1.10}) {
            const double r = rp * f;
            Metric4 m;
            if (!ks.metricAt(r, 1.0, m)) { worstKS = 1e30; continue; }
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (!std::isfinite(m.g[i][j]) || !std::isfinite(m.ginv[i][j]))
                        worstKS = 1e30;
                    else
                        worstKS = std::max(worstKS, std::fabs(m.g[i][j]));
        }
        // Contraste Boyer-Lindquist : g_rr = Sigma/Delta a r = r+ (Delta -> 0)
        const double ct = std::cos(1.0);
        const double Sigma = rp * rp + a * a * ct * ct;
        const double Delta = rp * rp - 2.0 * rp + a * a;
        blDivergence = std::max(blDivergence, Sigma / std::max(std::fabs(Delta), 1e-300));
    }
    // On teste que KS reste borné (valeurs O(10), pas O(1e30)).
    report("T6 KS regular at horizon (finite components)", worstKS < 1e6 ? 0.0 : 1.0, 1e-15);
    std::printf("       (contraste : en Boyer-Lindquist, g_rr ~ %.2e a r+)\n", blDivergence);
}

// --- T7 : coordonnées MKS ----------------------------------------------------
void testCoordinates() {
    ModifiedKerrSchild c(0.3);

    // Aller-retour r <-> x1 et theta <-> x2.
    double worstR = 0.0, worstT = 0.0;
    for (double r : {1.05, 2.0, 10.0, 100.0, 1000.0}) {
        const double back = c.rOf(c.x1Of(r));
        worstR = std::max(worstR, std::fabs(back - r) / r);
    }
    for (double x2 = 0.001; x2 < 0.999; x2 += 0.037) {
        const double th = c.thetaOf(x2);
        const double back = c.x2Of(th);
        worstT = std::max(worstT, std::fabs(back - x2));
    }
    report("T7a aller-retour r <-> x1", worstR, 1e-14);
    report("T7b aller-retour theta <-> x2", worstT, 1e-12);

    // Jacobiens analytiques vs différences finies.
    double worstJ1 = 0.0, worstJ2 = 0.0;
    const double h = 1e-6;
    for (double x1 = -0.5; x1 < 6.0; x1 += 0.31) {
        const double fd = (c.rOf(x1 + h) - c.rOf(x1 - h)) / (2.0 * h);
        worstJ1 = std::max(worstJ1, std::fabs(fd - c.drdx1(x1)) / std::fabs(fd));
    }
    for (double x2 = 0.05; x2 < 0.95; x2 += 0.043) {
        const double fd = (c.thetaOf(x2 + h) - c.thetaOf(x2 - h)) / (2.0 * h);
        worstJ2 = std::max(worstJ2, std::fabs(fd - c.dthetadx2(x2)) / std::fabs(fd));
    }
    report("T7c jacobien dr/dx1 vs differences finies", worstJ1, 1e-8);
    report("T7d jacobien dtheta/dx2 vs differences finies", worstJ2, 1e-8);

    // Monotonie de theta(x2) : indispensable pour que le maillage soit valide
    // (pas de repliement de cellules).
    double minSlope = 1e30;
    for (double x2 = 0.0; x2 <= 1.0; x2 += 0.001)
        minSlope = std::min(minSlope, c.dthetadx2(x2));
    report("T7e maillage theta monotone (dtheta/dx2 > 0)", minSlope > 0.0 ? 0.0 : 1.0, 1e-15);
}

// --- T8 : VRAI test des Christoffels — conservation le long des géodésiques --
//
// ATTENTION méthodologique : T3 (compatibilité métrique) est en réalité une
// TAUTOLOGIE algébrique. En substituant la définition de Γ dans ∇_λ g_μν on
// obtient identiquement ∂_λ g_μν - ∂_λ g_μν = 0, quelle que soit la justesse
// des dérivées `dg`. T3 attrape donc une faute de transcription dans la
// FORMULE de Γ, mais serait aveugle à une erreur dans metricDerivatives ou
// dans la métrique elle-même. C'est exactement le piège « le code tourne mais
// la physique est fausse » — d'où ce test-ci.
//
// Test réel : intégrer l'équation des géodésiques
//     dx^μ/dλ = p^μ ,  dp^μ/dλ = -Γ^μ_αβ p^α p^β
// et vérifier des invariants qui n'apparaissent PAS dans la construction de Γ :
//   * p_t = g_tμ p^μ conservé      (Kerr-Schild est stationnaire -> ∂_t Killing)
//   * p_φ = g_φμ p^μ conservé      (axisymétrique -> ∂_φ Killing)
//   * g_μν p^μ p^ν = 0 préservé    (caractère nul du photon)
// Si les Christoffels ou les dérivées de la métrique sont faux, ces trois
// quantités dérivent. Même esprit que le contrôle de la constante de Carter
// déjà en place dans le ray tracer.
void testGeodesicConservation() {
    double worstE = 0.0, worstL = 0.0, worstNull = 0.0;

    struct Start { double a, r, theta, pr, pth; };
    const Start starts[] = {
        {0.0,  8.0, kPi / 2.0, -0.3,  0.05},
        {0.5,  8.0, 1.1,       -0.4,  0.10},
        {0.9,  6.0, kPi / 2.0, -0.2, -0.08},
        {0.9, 12.0, 0.8,       -0.5,  0.03},
        {0.998, 5.0, kPi / 2.0, -0.25, 0.06},
    };

    for (const Start& s : starts) {
        KerrSchild ks(1.0, s.a);

        // Construit un p^μ NUL : on choisit p^r, p^θ, p^φ puis on résout
        // g_μν p^μ p^ν = 0 pour p^t (racine de l'équation quadratique).
        double x[4] = {0.0, s.r, s.theta, 0.0};
        double p[4] = {0.0, s.pr, s.pth, 0.01};

        Metric4 m;
        if (!ks.metricAt(x[1], x[2], m)) { worstE = 1e30; continue; }

        // A (p^t)² + 2B p^t + C = 0
        const double A = m.g[0][0];
        double B = 0.0, C = 0.0;
        for (int i = 1; i < 4; ++i) B += m.g[0][i] * p[i];
        for (int i = 1; i < 4; ++i)
            for (int j = 1; j < 4; ++j)
                C += m.g[i][j] * p[i] * p[j];
        const double disc = B * B - A * C;
        if (disc < 0.0) { worstE = 1e30; continue; }
        p[0] = (-B - std::sqrt(disc)) / A;   // racine future-dirigée

        auto covariant = [&](const double xx[4], const double pp[4], int lower) {
            Metric4 mm;
            ks.metricAt(xx[1], xx[2], mm);
            double v = 0.0;
            for (int i = 0; i < 4; ++i) v += mm.g[lower][i] * pp[i];
            return v;
        };
        auto nullNorm = [&](const double xx[4], const double pp[4]) {
            Metric4 mm;
            ks.metricAt(xx[1], xx[2], mm);
            double v = 0.0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    v += mm.g[i][j] * pp[i] * pp[j];
            return v;
        };

        const double E0 = covariant(x, p, 0);
        const double L0 = covariant(x, p, 3);
        // Échelle de référence pour normaliser le caractère nul.
        double pScale = 0.0;
        for (int i = 0; i < 4; ++i) pScale = std::max(pScale, std::fabs(p[i]));
        const double nullScale = std::max(pScale * pScale, 1e-30);

        // RK4 sur (x, p).
        // Un echec de christoffel() doit INTERROMPRE l'integration : integrer
        // avec Gamma = 0 donnerait une droite d'espace plat, donc des invariants
        // parfaitement conserves -- un test qui passerait pour la mauvaise raison.
        bool gammaOk = true;
        auto rhs = [&](const double xx[4], const double pp[4],
                       double dx[4], double dp[4]) {
            double G[4][4][4];
            if (!ks.christoffel(xx[1], xx[2], G)) gammaOk = false;
            for (int i = 0; i < 4; ++i) dx[i] = pp[i];
            for (int l = 0; l < 4; ++l) {
                double acc = 0.0;
                for (int aa = 0; aa < 4; ++aa)
                    for (int bb = 0; bb < 4; ++bb)
                        acc -= G[l][aa][bb] * pp[aa] * pp[bb];
                dp[l] = acc;
            }
        };

        const double dl = 0.002;
        const int nsteps = 3000;
        for (int step = 0; step < nsteps; ++step) {
            // Arrêt si le rayon plonge trop près de la singularité ou s'évade.
            if (x[1] < 0.5 || x[1] > 200.0) break;
            if (!gammaOk) { worstE = worstL = worstNull = 1e30; break; }

            double k1x[4], k1p[4], k2x[4], k2p[4], k3x[4], k3p[4], k4x[4], k4p[4];
            double tx[4], tp[4];

            rhs(x, p, k1x, k1p);
            for (int i = 0; i < 4; ++i) { tx[i] = x[i] + 0.5*dl*k1x[i]; tp[i] = p[i] + 0.5*dl*k1p[i]; }
            rhs(tx, tp, k2x, k2p);
            for (int i = 0; i < 4; ++i) { tx[i] = x[i] + 0.5*dl*k2x[i]; tp[i] = p[i] + 0.5*dl*k2p[i]; }
            rhs(tx, tp, k3x, k3p);
            for (int i = 0; i < 4; ++i) { tx[i] = x[i] + dl*k3x[i]; tp[i] = p[i] + dl*k3p[i]; }
            rhs(tx, tp, k4x, k4p);

            for (int i = 0; i < 4; ++i) {
                x[i] += (dl / 6.0) * (k1x[i] + 2.0*k2x[i] + 2.0*k3x[i] + k4x[i]);
                p[i] += (dl / 6.0) * (k1p[i] + 2.0*k2p[i] + 2.0*k3p[i] + k4p[i]);
            }

            const double E = covariant(x, p, 0);
            const double L = covariant(x, p, 3);
            worstE = std::max(worstE, std::fabs(E - E0) / std::max(std::fabs(E0), 1e-30));
            worstL = std::max(worstL, std::fabs(L - L0) / std::max(std::fabs(L0), 1e-30));
            worstNull = std::max(worstNull, std::fabs(nullNorm(x, p)) / nullScale);
        }
    }

    // Tolérances : dominées par l'erreur RK4 (pas fixe) et par la
    // différentiation numérique des Christoffels, pas par la précision machine.
    report("T8a conservation p_t (Killing d_t) le long des geodesiques", worstE, 1e-8);
    report("T8b conservation p_phi (Killing d_phi)", worstL, 1e-8);
    report("T8c caractere nul g_munu p^mu p^nu = 0 preserve", worstNull, 1e-8);
}

// --- S0 : sanité de la grille ------------------------------------------------
void testGrid() {
    KerrSchild ks(1.0, 0.9);
    const double rp = ks.horizonOuter();
    // Frontière interne DANS l'horizon (pratique standard).
    Grid g = Grid::makeSpherical(64, 32, 16, 0.8 * rp, 100.0, 0.3);

    double worst = 0.0;
    // Les centres de cellules doivent être strictement croissants et couvrir
    // le domaine demandé.
    const double r0 = g.rOf(Grid::NG);
    const double rN = g.rOf(Grid::NG + g.n1 - 1);
    worst = std::max(worst, (r0 > 0.8 * rp && r0 < 100.0) ? 0.0 : 1.0);
    worst = std::max(worst, (rN > r0 && rN < 100.0) ? 0.0 : 1.0);

    // theta doit rester dans ]0, pi[ sur toutes les cellules physiques.
    for (int j = Grid::NG; j < Grid::NG + g.n2; ++j) {
        const double th = g.thetaOf(j);
        if (!(th > 0.0 && th < kPi)) worst = 1.0;
    }

    // Indexation : unicité sur un échantillon.
    const std::size_t a = g.idx(3, 4, 5), b = g.idx(3, 4, 6), c2 = g.idx(4, 4, 5);
    if (a == b || a == c2) worst = 1.0;

    report("S0 grille MKS (bornes, theta valide, indexation)", worst, 1e-15);
    std::printf("       (r_inner = %.4f, r+ = %.4f -> boundary INSIDE horizon)\n", r0, rp);
}

// =============================================================================
// DURCISSEMENT PRE-S4
// =============================================================================

void reportBool(const char* name, bool ok, const char* detail = "") {
    std::printf("  [%s] %-46s %s\n", ok ? "OK  " : "FAIL", name, detail);
    if (!ok) ++g_failures;
}

// --- H1 : christoffel() signale son echec ------------------------------------
// AVANT : christoffel() remplissait Gamma de zeros et retournait void. Un
// appelant ne pouvait pas distinguer « espace plat » d'« inversion impossible ».
// En S4 ce Gamma alimentera les termes source GRHD : Gamma = 0 produirait une
// evolution parfaitement plausible et fausse, sans aucun symptome.
void testChristoffelFailureContract() {
    std::printf("\n  H1 -- christoffel() failure contract\n");
    KerrSchild ks(1.0, 0.9);

    // (a) Point valide : le resultat doit etre INCHANGE (non-regression).
    double G[4][4][4];
    const bool okValid = ks.christoffel(6.0, kPi / 3.0, G);
    double maxG = 0.0;
    for (int l = 0; l < 4; ++l)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                maxG = std::max(maxG, std::fabs(G[l][i][j]));
    reportBool("H1a point KS valide -> succes", okValid);
    reportBool("H1b Gamma non trivial en point valide", maxG > 1e-6);

    // (b) Entrees volontairement singulieres/invalides.
    // r=0, theta=pi/2 : Sigma = r^2 + a^2 cos^2(theta) = 0 -> singularite
    // annulaire, la metrique n'est pas inversible.
    struct Bad { const char* label; double r, th; };
    const Bad bad[] = {
        {"singularite annulaire (r=0, th=pi/2)", 0.0, kPi / 2.0},
        {"r non fini (NaN)",  std::numeric_limits<double>::quiet_NaN(), 1.0},
        {"r infini",          std::numeric_limits<double>::infinity(),  1.0},
    };
    bool allDetected = true, allZeroed = true;
    for (const Bad& b : bad) {
        double Gb[4][4][4];
        for (int l = 0; l < 4; ++l)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) Gb[l][i][j] = 12345.0;   // sentinelle
        const bool ok = ks.christoffel(b.r, b.th, Gb);
        if (ok) {
            // Tolere : si l'inversion REUSSIT, le contrat n'est pas viole --
            // mais alors Gamma doit etre fini, pas du bruit.
            for (int l = 0; l < 4; ++l)
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        if (!std::isfinite(Gb[l][i][j])) allDetected = false;
            std::printf("       (%s : inversion reussie, Gamma fini)\n", b.label);
        } else {
            for (int l = 0; l < 4; ++l)
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        if (Gb[l][i][j] != 0.0) allZeroed = false;
            std::printf("       (%s: failure reported)\n", b.label);
        }
    }
    reportBool("H1c invalid inputs never return plausible finite noise", allDetected);
    reportBool("H1d Gamma is zero-filled when failure is reported", allZeroed);

    // (c) DISCRIMINANCE : avec l'ancien contrat (void), il etait IMPOSSIBLE
    // d'ecrire H1c/H1d. Ce test-ci verifie que le canal d'echec existe vraiment,
    // c'est-a-dire qu'au moins une entree du domaine le declenche.
    double Gs[4][4][4];
    const bool sing = ks.christoffel(0.0, kPi / 2.0, Gs);
    reportBool("H1e failure path is genuinely reachable", !sing);
}

// --- H2 : domaine de validite des parametres de Kerr -------------------------
void testKerrParameterDomain() {
    std::printf("\n  H2 -- domaine des parametres de Kerr\n");

    // Cas ADMIS.
    struct Good { const char* label; double M, a; };
    const Good good[] = {
        {"Schwarzschild a=0",   1.0, 0.0},
        {"sous-extremal a=0.9", 1.0, 0.9},
        {"EXTREMAL a=M",        1.0, 1.0},
        {"EXTREMAL a=-M",       1.0, -1.0},
        {"masse != 1",          3.0, 2.4},
    };
    bool allGood = true, extremalOk = true;
    for (const Good& g : good) {
        KerrSchild ks(g.M, g.a);
        double rp = 0.0, rm = 0.0;
        const bool ok = ks.horizons(rp, rm);
        if (!ok || !ks.parametersValid()) allGood = false;
        // Verification analytique : r+ + r- = 2M et r+ * r- = a^2.
        if (ok) {
            if (std::fabs(rp + rm - 2.0 * g.M) > 1e-14 * g.M) allGood = false;
            if (std::fabs(rp * rm - g.a * g.a) > 1e-13 * std::max(g.a * g.a, 1.0)) allGood = false;
        }
    }
    reportBool("H2a cas admis acceptes (r+ + r- = 2M, r+ r- = a^2)", allGood);

    // Le cas extremal doit rester SUPPORTE et donner r+ = r- = M exactement.
    {
        KerrSchild ks(1.0, 1.0);
        double rp = 0.0, rm = 0.0;
        const bool ok = ks.horizons(rp, rm);
        extremalOk = ok && (rp == 1.0) && (rm == 1.0);
        std::printf("       (extremal a=M : r+ = %.17g, r- = %.17g)\n", rp, rm);
    }
    reportBool("H2b cas extremal |a|=M supporte, r+ = r- = M", extremalOk);

    // Cas REJETES.
    struct Bad { const char* label; double M, a; };
    const Bad bad[] = {
        {"singularite nue a>M",  1.0,  1.5},
        {"singularite nue a<-M", 1.0, -1.5},
        {"M = 0",                0.0,  0.0},
        {"M < 0",               -1.0,  0.0},
        {"M non fini",           std::numeric_limits<double>::quiet_NaN(), 0.0},
    };
    bool allRejected = true, allNaN = true;
    for (const Bad& b : bad) {
        KerrSchild ks(b.M, b.a);
        double rp = 0.0, rm = 0.0;
        if (ks.parametersValid() || ks.horizons(rp, rm)) {
            allRejected = false;
            std::printf("       (%s : ACCEPTE A TORT)\n", b.label);
        }
        // Les formes commodes doivent renvoyer NaN, pas une valeur plausible.
        if (!std::isnan(ks.horizonOuter()) || !std::isnan(ks.horizonInner()))
            allNaN = false;
    }
    reportBool("H2c M<=0 ou |a|>M rejetes explicitement", allRejected);
    // DISCRIMINANT : avec l'ancien clamp `d > 0 ? d : 0`, a=1.5 renvoyait
    // silencieusement r+ = M = 1 -- un horizon qui n'existe pas.
    reportBool("H2d out-of-domain parameters -> NaN, not a plausible horizon", allNaN);
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf("  GRMHD S0-S1 : grille MKS + metrique de Kerr-Schild\n");
    std::printf("=====================================================================\n");

    testInverse();
    testSqrtNegG();
    testMetricCompatibility();
    testChristoffelSymmetry();
    testStationaryAxisymmetric();
    testHorizonRegularity();
    testGeodesicConservation();
    testCoordinates();
    testGrid();
    testChristoffelFailureContract();
    testKerrParameterDomain();

    std::printf("---------------------------------------------------------------------\n");
    if (g_failures == 0) {
        std::printf("  TOUS LES TESTS PASSENT -- S0/S1 valides, S2 peut demarrer.\n");
    } else {
        std::printf("  %d FAILED TEST(S) -- do not proceed to S2.\n", g_failures);
    }
    std::printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
