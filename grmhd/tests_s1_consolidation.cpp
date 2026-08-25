// -----------------------------------------------------------------------------
// S1 — CONSOLIDATION avant de bâtir tout le solveur sur cette géométrie.
//
// Deux blocs :
//
//  A. VÉRIFICATION DU SCHÉMA DE DIFFÉRENTIATION des Christoffels
//     Les tests de S1 validaient les *conséquences* (géodésiques). Ici on
//     valide le *calcul des dérivées lui-même*, contre une référence
//     indépendante à la précision machine : la différentiation par PAS
//     COMPLEXE  f'(x) = Im[f(x+ih)]/h + O(h²).
//     Contrairement aux différences finies, elle ne fait aucune soustraction
//     de valeurs proches : pas d'annulation catastrophique, donc on peut
//     prendre h = 1e-200 et obtenir la dérivée exacte aux erreurs d'arrondi
//     près. C'est un oracle authentiquement indépendant du schéma testé.
//
//  B. VALIDATION CROISÉE Kerr-Schild ↔ Boyer-Lindquist
//     Le ray tracer BL de blackholesim (src/Metric.cpp, src/Geodesic.cpp) est
//     une implémentation INDÉPENDANTE, déjà validée en production par la
//     conservation de la constante de Carter (~1e-8). On l'utilise comme
//     oracle externe : mêmes géodésiques physiques, deux formalismes
//     (hamiltonien canonique en BL vs équation des géodésiques avec
//     Christoffels en KS), et on exige que l'écart CONVERGE à l'ordre du
//     schéma quand le pas diminue.
//
//     Transformation BL -> KS (mêmes r et θ ; seuls t et φ diffèrent) :
//        dt_KS = dt_BL + (2Mr/Δ) dr ,  dφ_KS = dφ_BL + (a/Δ) dr
//     d'où pour un vecteur contravariant :
//        p^t_KS = p^t_BL + (2Mr/Δ) p^r ,  p^φ_KS = p^φ_BL + (a/Δ) p^r
//        p^r et p^θ sont INCHANGÉS  -> comparables directement, sans
//        transformation, ce qui élimine tout risque d'erreur de conversion
//        dans le test lui-même.
//     Δ → 0 à l'horizon : la transformation y est singulière (c'est la raison
//     d'être de KS), donc BL ne peut servir d'oracle qu'à l'extérieur.
// -----------------------------------------------------------------------------
#include "KerrSchild.hpp"
#include "Coordinates.hpp"

#include "Metric.hpp"      // bhsim::Kerr        (oracle BL)
#include "Geodesic.hpp"    // bhsim::geodesicRHS (oracle BL)

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace grmhd;

namespace {

int g_failures = 0;

void report(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  [%s] %-52s  err = %.3e  (tol %.1e)\n",
                ok ? "OK  " : "FAIL", name, err, tol);
    if (!ok) ++g_failures;
}

// =============================================================================
//  BLOC A — schéma de différentiation
// =============================================================================

using cplx = std::complex<double>;

// Dérivées ∂_r g_μν et ∂_θ g_μν par pas complexe (référence exacte).
void metricDerivComplexStep(double r, double theta, double M, double a,
                            double dgdr[4][4], double dgdth[4][4]) {
    const double h = 1e-200;   // aucun risque d'annulation : pas de soustraction

    {   // ∂/∂r : r -> r + ih
        cplx gc[4][4];
        const cplx rr(r, h);
        const cplx st(std::sin(theta), 0.0), ct(std::cos(theta), 0.0);
        metricFromTrig<cplx>(rr, st, ct, cplx(M, 0.0), cplx(a, 0.0), gc);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                dgdr[i][j] = gc[i][j].imag() / h;
    }
    {   // ∂/∂θ : θ -> θ + ih, propagé analytiquement dans sin/cos complexes
        cplx gc[4][4];
        const cplx th(theta, h);
        const cplx st = std::sin(th), ct = std::cos(th);
        metricFromTrig<cplx>(cplx(r, 0.0), st, ct, cplx(M, 0.0), cplx(a, 0.0), gc);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                dgdth[i][j] = gc[i][j].imag() / h;
    }
}

// A1 — les dérivées finies centrées du module vs le pas complexe.
// Valide DIRECTEMENT le calcul des dérivées, pas seulement ses conséquences.
void testDerivativesVsComplexStep() {
    double worstR = 0.0, worstT = 0.0;
    const double spins[]  = {0.0, 0.5, 0.9, 0.998};
    const double radii[]  = {1.2, 2.0, 5.0, 20.0, 200.0};
    // On reste à distance raisonnable de l'axe : voir A3 pour son traitement.
    const double thetas[] = {0.25, 0.8, kPi / 2.0, 2.3, kPi - 0.25};

    for (double a : spins) {
        KerrSchild ks(1.0, a);
        for (double r : radii) {
            for (double th : thetas) {
                double dg[4][4][4];
                ks.metricDerivatives(r, th, dg);

                double refR[4][4], refT[4][4];
                metricDerivComplexStep(r, th, 1.0, a, refR, refT);

                double scale = 0.0;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        scale = std::max(scale, std::max(std::fabs(refR[i][j]),
                                                          std::fabs(refT[i][j])));
                scale = std::max(scale, 1e-30);

                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        worstR = std::max(worstR, std::fabs(dg[1][i][j] - refR[i][j]) / scale);
                        worstT = std::max(worstT, std::fabs(dg[2][i][j] - refT[i][j]) / scale);
                    }
                }
            }
        }
    }
    // Différences centrées O(h²) avec h~1e-6 => ~1e-12 attendu, plus l'arrondi.
    report("A1a d_r g : differences finies vs pas complexe", worstR, 1e-8);
    report("A1b d_th g : differences finies vs pas complexe", worstT, 1e-8);
}

// A2 — sensibilité des Γ au pas h : balayage sur plusieurs décades.
// Doit exhiber la courbe en V classique (troncature O(h²) qui décroît,
// arrondi O(eps/h) qui croît) avec un plateau autour de l'optimum ~eps^(1/3).
// Le but est de VÉRIFIER que le h retenu (1e-6) est dans le plateau, pas
// d'obtenir un chiffre magique.
void testStepSensitivity() {
    const double r = 4.0, th = 1.0, a = 0.9;
    double refR[4][4], refT[4][4];
    metricDerivComplexStep(r, th, 1.0, a, refR, refT);

    std::printf("       balayage du pas h (derivee d_r g, spin 0.9, r=4, th=1) :\n");
    double bestErr = 1e30; double bestH = 0.0;
    double errAtProduction = 0.0;
    for (double h = 1e-2; h > 1e-12; h *= 0.1) {
        double gp[4][4], gm[4][4];
        KerrSchild ks(1.0, a);
        ks.metricCovariant(r + h, th, gp);
        ks.metricCovariant(r - h, th, gm);
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                scale = std::max(scale, std::fabs(refR[i][j]));
        scale = std::max(scale, 1e-30);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                worst = std::max(worst, std::fabs((gp[i][j] - gm[i][j]) / (2.0 * h) - refR[i][j]) / scale);
        std::printf("         h = %.0e  ->  err = %.3e\n", h, worst);
        if (worst < bestErr) { bestErr = worst; bestH = h; }
        // h de production : 1e-6 * max(|r|,1) = 4e-6 ici ; on relève 1e-6.
        if (std::fabs(h - 1e-6) < 1e-18) errAtProduction = worst;
    }
    std::printf("       optimum empirique : h = %.0e (err %.2e) ; h de production = 1e-6\n",
                bestH, bestErr);
    // Exigence : le h de production doit être à moins de ~3 ordres de
    // grandeur de l'optimum — c'est-à-dire dans le plateau, pas sur un flanc.
    report("A2 h de production dans le plateau du balayage",
           errAtProduction / std::max(bestErr, 1e-30), 1e3);
}

// A3 — comportement près de l'axe polaire.
//
// ⚠️ DIAGNOSTIC INITIAL ERRONÉ, CORRIGÉ APRÈS PEER REVIEW (2026-08-25).
// La première version affirmait que le franchissement de l'axe par θ-h
// provoquait une « réflexion non analytique » via |sinθ|, et en déduisait une
// contrainte de maillage θ ≥ 1e-2 rad. C'est FAUX, pour deux raisons :
//
//  (1) Dans metricFromTrig, `st` n'intervient QUE par `s2 = st*st`, et `ct`
//      que par `ct*ct`. Le carré efface la valeur absolue :
//          (|sin θ|)² = sin²θ  exactement.
//      Les composantes de g sont donc des fonctions de sin²θ et cos²θ, soit
//      des fonctions PAIRES et analytiques de θ autour de 0. Franchir l'axe
//      ne produit aucune réflexion : g(-θ) = g(θ) est la bonne prolongation.
//      (Le plancher 1e-14 ne s'active d'ailleurs que pour θ < 1e-14.)
//
//  (2) g étant paire, ∂_θ g est IMPAIRE, donc ∂_θ g → 0 sur l'axe. L'ancien
//      test normalisait l'erreur absolue par max|∂_θ g_ref| — une référence
//      qui s'annule. La « dégradation » observée n'était donc qu'un problème
//      de CONDITIONNEMENT DU DIAGNOSTIC, pas du schéma.
//
// Ce test mesure maintenant les trois quantités séparément pour trancher sur
// la donnée : erreur absolue, erreur relative à la référence évanescente
// (mal conditionnée), et erreur relative à une échelle métrique qui ne
// s'annule PAS (max|g|, dimensionnellement homogène puisque θ est sans
// dimension). Puis il teste ce qui compte réellement pour le solveur : les
// Christoffels.
void testAxisProximity() {
    const double a = 0.9, r = 4.0;
    KerrSchild ks(1.0, a);

    // --- A3a : g est-elle PAIRE autour de l'axe ? (réfute la réflexion) -----
    double worstParity = 0.0;
    for (double th : {1e-7, 1e-5, 1e-3, 1e-2, 0.1}) {
        double gp[4][4], gm[4][4];
        ks.metricCovariant(r, th, gp);
        ks.metricCovariant(r, -th, gm);
        double scale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) scale = std::max(scale, std::fabs(gp[i][j]));
        scale = std::max(scale, 1e-30);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                worstParity = std::max(worstParity, std::fabs(gp[i][j] - gm[i][j]) / scale);
    }
    report("A3a g(-theta) == g(theta) : pas de reflexion a l'axe", worstParity, 1e-15);

    // --- A3b : les trois métriques d'erreur, cote à cote --------------------
    std::printf("       d_th g pres de l'axe -- trois normalisations :\n");
    std::printf("         %-10s %-12s %-14s %-14s %s\n",
                "theta", "|err| abs", "rel/|d_th g|", "rel/max|g|", "max|d_th g_ref|");
    double worstAbs = 0.0, worstVsG = 0.0;
    for (double th : {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}) {
        double dg[4][4][4];
        ks.metricDerivatives(r, th, dg);
        double refR[4][4], refT[4][4];
        metricDerivComplexStep(r, th, 1.0, a, refR, refT);
        double gg[4][4];
        ks.metricCovariant(r, th, gg);

        double absErr = 0.0, refScale = 0.0, gScale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                absErr   = std::max(absErr, std::fabs(dg[2][i][j] - refT[i][j]));
                refScale = std::max(refScale, std::fabs(refT[i][j]));
                gScale   = std::max(gScale, std::fabs(gg[i][j]));
            }
        const double relRef = absErr / std::max(refScale, 1e-300);
        const double relG   = absErr / std::max(gScale, 1e-300);
        std::printf("         %-10.0e %-12.3e %-14.3e %-14.3e %.3e\n",
                    th, absErr, relRef, relG, refScale);
        worstAbs = std::max(worstAbs, absErr);
        worstVsG = std::max(worstVsG, relG);
    }
    // VERDICT attendu si l'objection est correcte : |err| abs et rel/max|g|
    // restent ~plats sur 6 décades, tandis que rel/|d_th g| explose parce que
    // son dénominateur tend vers 0.
    report("A3b erreur ABSOLUE de d_th g bornee jusqu'a l'axe", worstAbs, 1e-7);
    report("A3c erreur relative a max|g| (echelle non evanescente)", worstVsG, 1e-9);

    // --- A3d : ce qui compte vraiment pour le solveur, les Christoffels -----
    // Γ ~ g⁻¹ ∂g, or g^{φφ} ~ 1/(Σ sin²θ) diverge sur l'axe : c'est LÀ que se
    // poserait un vrai problème de conditionnement, pas dans ∂g. On compare
    // les Γ issus des différences finies à ceux issus du pas complexe (exact).
    std::printf("       erreur relative sur les Christoffels pres de l'axe :\n");
    double worstGamma = 0.0;
    for (double th : {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5}) {
        Metric4 m;
        if (!ks.metricAt(r, th, m)) continue;

        double refR[4][4], refT[4][4];
        metricDerivComplexStep(r, th, 1.0, a, refR, refT);
        double dgRef[4][4][4] = {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) { dgRef[1][i][j] = refR[i][j]; dgRef[2][i][j] = refT[i][j]; }

        double Gref[4][4][4];
        for (int l = 0; l < 4; ++l)
            for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu) {
                    double s = 0.0;
                    for (int sg = 0; sg < 4; ++sg)
                        s += m.ginv[l][sg] * (dgRef[mu][sg][nu] + dgRef[nu][sg][mu] - dgRef[sg][mu][nu]);
                    Gref[l][mu][nu] = 0.5 * s;
                }

        double Gfd[4][4][4];
        ks.christoffel(r, th, Gfd);

        double err = 0.0, scale = 0.0;
        for (int l = 0; l < 4; ++l)
            for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu) {
                    err   = std::max(err, std::fabs(Gfd[l][mu][nu] - Gref[l][mu][nu]));
                    scale = std::max(scale, std::fabs(Gref[l][mu][nu]));
                }
        const double rel = err / std::max(scale, 1e-300);
        std::printf("         theta = %-8.0e  err_rel(Gamma) = %.3e   (max|Gamma| = %.3e)\n",
                    th, rel, scale);
        worstGamma = std::max(worstGamma, rel);
    }
    report("A3d Christoffels : erreur relative bornee jusqu'a 1e-5 rad", worstGamma, 1e-8);
}

// =============================================================================
//  BLOC B — validation croisée KS <-> BL
// =============================================================================

struct GeoIC {
    double a;          // spin
    double r0, th0;    // position initiale
    double pr, pth;    // composantes contravariantes p^r, p^θ
    double pphi;       // p^φ (en KS)
    const char* label;
};

// Intègre en KS avec l'équation des géodésiques + Christoffels (code NEUF),
// pas fixe RK4. Renvoie l'état final et les dérives d'invariants.
struct KSResult { double r, th, pr, pth; double dE, dL, dNull; bool ok; };

KSResult integrateKS(const KerrSchild& ks, const GeoIC& ic, double dl, int nsteps) {
    KSResult out{};
    double x[4] = {0.0, ic.r0, ic.th0, 0.0};
    double p[4] = {0.0, ic.pr, ic.pth, ic.pphi};

    Metric4 m;
    if (!ks.metricAt(x[1], x[2], m)) { out.ok = false; return out; }
    // p^t par la condition de nullité g_μν p^μ p^ν = 0.
    const double A = m.g[0][0];
    double B = 0.0, C = 0.0;
    for (int i = 1; i < 4; ++i) B += m.g[0][i] * p[i];
    for (int i = 1; i < 4; ++i)
        for (int j = 1; j < 4; ++j) C += m.g[i][j] * p[i] * p[j];
    const double disc = B * B - A * C;
    if (disc < 0.0) { out.ok = false; return out; }
    p[0] = (-B - std::sqrt(disc)) / A;

    auto cov = [&](const double xx[4], const double pp[4], int lo) {
        Metric4 mm; ks.metricAt(xx[1], xx[2], mm);
        double v = 0.0; for (int i = 0; i < 4; ++i) v += mm.g[lo][i] * pp[i];
        return v;
    };
    auto nrm = [&](const double xx[4], const double pp[4]) {
        Metric4 mm; ks.metricAt(xx[1], xx[2], mm);
        double v = 0.0;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) v += mm.g[i][j] * pp[i] * pp[j];
        return v;
    };

    const double E0 = cov(x, p, 0), L0 = cov(x, p, 3);
    double pScale = 0.0;
    for (int i = 0; i < 4; ++i) pScale = std::max(pScale, std::fabs(p[i]));
    const double nullScale = std::max(pScale * pScale, 1e-30);

    auto rhs = [&](const double xx[4], const double pp[4], double dx[4], double dp[4]) {
        double G[4][4][4];
        ks.christoffel(xx[1], xx[2], G);
        for (int i = 0; i < 4; ++i) dx[i] = pp[i];
        for (int l = 0; l < 4; ++l) {
            double acc = 0.0;
            for (int aa = 0; aa < 4; ++aa)
                for (int bb = 0; bb < 4; ++bb)
                    acc -= G[l][aa][bb] * pp[aa] * pp[bb];
            dp[l] = acc;
        }
    };

    for (int s = 0; s < nsteps; ++s) {
        double k1x[4],k1p[4],k2x[4],k2p[4],k3x[4],k3p[4],k4x[4],k4p[4],tx[4],tp[4];
        rhs(x,p,k1x,k1p);
        for (int i=0;i<4;++i){tx[i]=x[i]+0.5*dl*k1x[i];tp[i]=p[i]+0.5*dl*k1p[i];}
        rhs(tx,tp,k2x,k2p);
        for (int i=0;i<4;++i){tx[i]=x[i]+0.5*dl*k2x[i];tp[i]=p[i]+0.5*dl*k2p[i];}
        rhs(tx,tp,k3x,k3p);
        for (int i=0;i<4;++i){tx[i]=x[i]+dl*k3x[i];tp[i]=p[i]+dl*k3p[i];}
        rhs(tx,tp,k4x,k4p);
        for (int i=0;i<4;++i){
            x[i]+=(dl/6.0)*(k1x[i]+2.0*k2x[i]+2.0*k3x[i]+k4x[i]);
            p[i]+=(dl/6.0)*(k1p[i]+2.0*k2p[i]+2.0*k3p[i]+k4p[i]);
        }
        out.dE = std::max(out.dE, std::fabs(cov(x,p,0)-E0)/std::max(std::fabs(E0),1e-30));
        out.dL = std::max(out.dL, std::fabs(cov(x,p,3)-L0)/std::max(std::fabs(L0),1e-30));
        out.dNull = std::max(out.dNull, std::fabs(nrm(x,p))/nullScale);
    }
    out.r = x[1]; out.th = x[2]; out.pr = p[1]; out.pth = p[2];
    out.ok = true;
    return out;
}

// Intègre la MÊME géodésique physique avec le code BL de blackholesim
// (formalisme hamiltonien canonique, implémentation totalement indépendante).
struct BLResult { double r, th, pr, pth; double dK; bool ok; };

BLResult integrateBL(const GeoIC& ic, double dl, int nsteps) {
    BLResult out{};
    const double M = 1.0, a = ic.a;
    const bhsim::Kerr bh(M, a);

    // Conversion KS -> BL du quadri-moment (r, θ inchangés).
    const double r = ic.r0, th = ic.th0;
    const double Delta = r * r - 2.0 * M * r + a * a;
    if (std::fabs(Delta) < 1e-12) { out.ok = false; return out; }

    // p^φ_BL = p^φ_KS - (a/Δ) p^r   (p^r, p^θ identiques)
    const double pphiBL = ic.pphi - (a / Delta) * ic.pr;

    // p^t_BL par la condition de nullité dans la métrique BL.
    const double st = std::sin(th), ct = std::cos(th);
    const double Sigma = r * r + a * a * ct * ct;
    const double s2 = st * st;
    // Métrique BL covariante.
    const double gtt = -(1.0 - 2.0 * M * r / Sigma);
    const double gtp = -2.0 * M * a * r * s2 / Sigma;
    const double grr = Sigma / Delta;
    const double gthth = Sigma;
    const double gpp = s2 * (r * r + a * a + 2.0 * M * a * a * r * s2 / Sigma);

    const double A = gtt;
    const double B = gtp * pphiBL;
    const double C = grr * ic.pr * ic.pr + gthth * ic.pth * ic.pth + gpp * pphiBL * pphiBL;
    const double disc = B * B - A * C;
    if (disc < 0.0) { out.ok = false; return out; }
    const double ptBL = (-B - std::sqrt(disc)) / A;

    // Constantes du mouvement (covariantes) attendues par le code bhsim.
    const double p_t   = gtt * ptBL + gtp * pphiBL;
    const double p_phi = gtp * ptBL + gpp * pphiBL;
    const double p_r   = grr * ic.pr;
    const double p_th  = gthth * ic.pth;

    bhsim::Constants c;
    c.E  = -p_t;
    c.Lz = p_phi;
    c.Q  = p_th * p_th + ct * ct * (c.Lz * c.Lz / std::max(s2, 1e-300) - a * a * c.E * c.E);
    const double LmaE = c.Lz - a * c.E;
    c.K = c.Q + LmaE * LmaE;

    bhsim::State y;
    y.r = r; y.theta = th; y.phi = 0.0; y.pr = p_r; y.pth = p_th;

    const double K0 = c.K;
    auto f = [&](const bhsim::State& s) { return bhsim::geodesicRHS(s, bh, c); };

    // ATTENTION : le code bhsim normalise E = 1 par convention interne dans
    // le renderer, mais geodesicRHS utilise c.E tel quel — on garde donc nos
    // valeurs physiques. L'affine parameter est le même (dx/dλ = g^{μν}p_ν).
    for (int s = 0; s < nsteps; ++s) {
        // RK4 à pas fixe (le même que côté KS) pour rendre la comparaison de
        // convergence significative : deux schémas d'ordre 4 sur le même λ.
        const bhsim::State k1 = f(y);
        bhsim::State t2 = y + (0.5 * dl) * k1; const bhsim::State k2 = f(t2);
        bhsim::State t3 = y + (0.5 * dl) * k2; const bhsim::State k3 = f(t3);
        bhsim::State t4 = y + dl * k3;         const bhsim::State k4 = f(t4);
        y = y + (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        const double drift = std::fabs(bhsim::carterK(y, bh, c) - K0) / (std::fabs(K0) + 1e-30);
        out.dK = std::max(out.dK, drift);
    }

    // Reconversion en composantes contravariantes pour comparer à KS.
    const double ctF = std::cos(y.theta), stF = std::sin(y.theta);
    const double SigmaF = y.r * y.r + a * a * ctF * ctF;
    const double DeltaF = y.r * y.r - 2.0 * M * y.r + a * a;
    out.r = y.r; out.th = y.theta;
    out.pr  = (DeltaF / SigmaF) * y.pr;    // p^r = g^rr p_r  en BL
    out.pth = y.pth / SigmaF;              // p^θ = g^θθ p_θ
    (void)stF;
    out.ok = true;
    return out;
}

// B1 — comparaison directe + B2 convergence.
void testCrossValidationKSvsBL() {
    const GeoIC ics[] = {
        {0.0,  20.0, kPi / 2.0, -0.35,  0.00,  0.010, "a=0    equatorial"},
        {0.5,  20.0, 1.10,      -0.35,  0.03,  0.008, "a=0.5  incline"},
        {0.9,  15.0, kPi / 2.0, -0.30,  0.00,  0.012, "a=0.9  equatorial"},
        {0.9,  12.0, 0.90,      -0.40,  0.04,  0.006, "a=0.9  incline"},
        {0.998, 8.0, kPi / 2.0, -0.25,  0.00,  0.015, "a=0.998 proche horizon"},
    };

    std::printf("       comparaison KS (Christoffels) vs BL (hamiltonien bhsim) :\n");
    double worstDiff = 0.0, worstCarter = 0.0, worstKSinv = 0.0;

    for (const GeoIC& ic : ics) {
        KerrSchild ks(1.0, ic.a);
        const double dl = 0.01;
        const int n = 2000;   // λ_final = 20

        const KSResult rk = integrateKS(ks, ic, dl, n);
        const BLResult rb = integrateBL(ic, dl, n);
        if (!rk.ok || !rb.ok) { worstDiff = 1e30; continue; }

        // r et θ sont des coordonnées PARTAGÉES entre BL et KS : comparaison
        // directe, sans aucune transformation (donc sans risque d'erreur
        // introduite par le test lui-même).
        const double er  = std::fabs(rk.r - rb.r) / std::max(std::fabs(rb.r), 1e-30);
        const double eth = std::fabs(rk.th - rb.th);
        const double epr = std::fabs(rk.pr - rb.pr) / std::max(std::fabs(rb.pr), 1e-30);

        std::printf("         %-24s r: %.3e  th: %.3e  p^r: %.3e  | Carter(BL) %.2e\n",
                    ic.label, er, eth, epr, rb.dK);
        worstDiff = std::max(worstDiff, std::max(er, eth));
        worstCarter = std::max(worstCarter, rb.dK);
        worstKSinv = std::max(worstKSinv, std::max(rk.dE, std::max(rk.dL, rk.dNull)));
    }

    report("B1a accord KS vs BL sur (r, theta) a lambda=20", worstDiff, 1e-6);
    report("B1b invariants KS (E, Lz, nullite) le long du trajet", worstKSinv, 1e-8);
    report("B1c Carter conservee cote BL (sanite de l'oracle)", worstCarter, 1e-8);
}

// B2 — CONVERGENCE : c'est le test décisif. En divisant le pas par 2, l'écart
// KS↔BL doit chuter d'un facteur ~16 (deux schémas RK4, ordre 4), jusqu'au
// plancher imposé par la différentiation numérique des Christoffels.
// Une égalité bit-à-bit n'est ni attendue ni demandée : c'est la LOI de
// décroissance qui prouve que les deux implémentations convergent vers la
// même solution physique.
void testConvergenceOrder() {
    const GeoIC ic = {0.9, 15.0, 1.05, -0.30, 0.03, 0.010, "convergence"};
    KerrSchild ks(1.0, ic.a);

    // DIMENSIONNEMENT DU TEST : une mesure d'ordre de convergence n'a de sens
    // que dans le régime où l'erreur de TRONCATURE domine. Un premier essai
    // partait de dl=0.02 : l'écart y valait déjà ~1e-12, c'est-à-dire le
    // plancher numérique (Christoffels par différences finies + arrondi
    // accumulé), et les « ordres » mesurés oscillaient (+2.7 / -2.2 / +2.3)
    // — on mesurait du bruit. On part donc de pas nettement plus grossiers,
    // et on n'évalue l'ordre que sur les paires dont les deux erreurs sont
    // franchement au-dessus du plancher.
    constexpr double kFloor = 1e-11;   // plancher numérique constaté

    std::printf("       convergence de l'ecart KS<->BL (lambda_final = 10 fixe) :\n");
    double prevErr = 0.0, prevDl = 0.0;
    double bestOrder = 0.0;
    bool measuredInTruncationRegime = false;

    for (int k = 0; k < 7; ++k) {
        const double dl = 0.8 / std::pow(2.0, k);
        const int n = static_cast<int>(std::lround(10.0 / dl));
        const KSResult rk = integrateKS(ks, ic, dl, n);
        const BLResult rb = integrateBL(ic, dl, n);
        if (!rk.ok || !rb.ok) continue;
        const double e = std::fabs(rk.r - rb.r) + std::fabs(rk.th - rb.th);

        std::printf("         dl = %.5f  ecart = %.3e", dl, e);
        if (k > 0 && prevErr > 0.0 && e > 0.0) {
            const double order = std::log2(prevErr / e);
            const bool bothAboveFloor = (e > 100.0 * kFloor) && (prevErr > 100.0 * kFloor);
            std::printf("   ordre = %+.2f %s", order,
                        bothAboveFloor ? "(regime troncature)" : "(plancher atteint)");
            if (bothAboveFloor) {
                bestOrder = std::max(bestOrder, order);
                measuredInTruncationRegime = true;
            }
        }
        std::printf("\n");
        prevErr = e; prevDl = dl;
    }
    (void)prevDl;

    report("B2a mesure effectuee dans le regime de troncature",
           measuredInTruncationRegime ? 0.0 : 1.0, 1e-15);
    report("B2b ordre de convergence observe >= 3",
           (bestOrder >= 3.0) ? 0.0 : 1.0, 1e-15);
    std::printf("       (ordre max observe hors plancher : %.2f ; le plateau final a\n"
                "        ~1e-12 vient du plancher de differentiation des Christoffels,\n"
                "        c'est attendu et non un echec)\n", bestOrder);
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf("  S1 CONSOLIDATION : schema de derivation + croisement KS <-> BL\n");
    std::printf("=====================================================================\n");
    std::printf("\n  BLOC A -- verification du schema de differentiation\n");
    testDerivativesVsComplexStep();
    testStepSensitivity();
    testAxisProximity();

    std::printf("\n  BLOC B -- validation croisee contre l'oracle Boyer-Lindquist\n");
    testCrossValidationKSvsBL();
    testConvergenceOrder();

    std::printf("---------------------------------------------------------------------\n");
    if (g_failures == 0)
        std::printf("  CONSOLIDATION S1 VALIDEE -- la geometrie est verrouillee, S2 peut demarrer.\n");
    else
        std::printf("  %d ECHEC(S) -- geometrie NON verrouillee, ne pas demarrer S2.\n", g_failures);
    std::printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
