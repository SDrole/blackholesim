// -----------------------------------------------------------------------------
// S3 — Validation du solveur SRHD 1D en espace plat.
//
// Ordre délibéré : chaque brique est testée SEULE avant d'être composée, pour
// qu'un échec désigne un coupable unique.
//   A. flux physiques        (avant toute reconstruction/évolution)
//   B. vitesses caractéristiques
//   C. HLLE isolé
//   D. conservation globale discrète
//   E. convergence sur solution LISSE (avant les chocs)
//   F. tubes à choc vs oracle de Riemann EXACT
//   G. invariance par boost de Lorentz
//   H. HDF5 (écriture + relecture)
// -----------------------------------------------------------------------------
#include "Solver1D.hpp"
#include "Riemann.hpp"
#include "Flux.hpp"
#include "ExactRiemannSRHD.hpp"
#include "Hdf5Writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace grmhd;

namespace {

int g_failures = 0;
int g_tests = 0;

void report(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    ++g_tests;
    std::printf("  [%s] %-52s  err = %.3e  (tol %.1e)\n",
                ok ? "OK  " : "FAIL", name, err, tol);
    if (!ok) ++g_failures;
}
void reportBool(const char* name, bool ok, const char* detail = "") {
    ++g_tests;
    std::printf("  [%s] %-52s  %s\n", ok ? "OK  " : "FAIL", name, detail);
    if (!ok) ++g_failures;
}

PrimitiveState mkPrim(double rho, double u, double vx, double vy = 0.0, double vz = 0.0) {
    PrimitiveState P; P.rho = rho; P.u = u;
    const double v2 = vx*vx + vy*vy + vz*vz;
    const double W = 1.0 / std::sqrt(1.0 - v2);
    P.utilde[0] = W * vx; P.utilde[1] = W * vy; P.utilde[2] = W * vz;
    return P;
}

// =============================================================================
//  A — FLUX PHYSIQUES (testés AVANT reconstruction et Riemann)
// =============================================================================
void testPhysicalFlux() {
    const IdealGasEOS eos(5.0 / 3.0);

    // A1 — fluide au repos : seul F_{S_x} = p doit être non nul.
    {
        const PrimitiveState P = mkPrim(2.0, 1.5, 0.0);
        const FluidVars f = makeFluidVars(P, eos);
        const ConservedState U = primitiveToConserved(P, eos);
        double F[5]; physicalFlux(f, U, 0, F);
        const double p = eos.pressure(P.rho, P.u);
        double worst = 0.0;
        worst = std::max(worst, std::fabs(F[0]));            // pas de flux de masse
        worst = std::max(worst, std::fabs(F[1] - p) / p);    // = pression
        worst = std::max(worst, std::fabs(F[2]));
        worst = std::max(worst, std::fabs(F[3]));
        worst = std::max(worst, std::fabs(F[4]));            // pas de flux d'énergie
        report("A1 fluide au repos : F = (0, p, 0, 0, 0)", worst, 1e-15);
    }

    // A2 — advection uniforme : le flux de masse doit valoir exactement D·v.
    {
        double worst = 0.0;
        for (double v : {0.1, 0.5, 0.9, 0.99}) {
            const PrimitiveState P = mkPrim(1.3, 0.7, v);
            const FluidVars f = makeFluidVars(P, eos);
            const ConservedState U = primitiveToConserved(P, eos);
            double F[5]; physicalFlux(f, U, 0, F);
            worst = std::max(worst, std::fabs(F[0] - U.D * v) / std::fabs(U.D * v));
        }
        report("A2 advection : F_D = D v exactement", worst, 1e-15);
    }

    // A3 — impulsion transverse : F_{S_y} = S_y v^x (pas de terme de pression
    // hors diagonale) et S_y est bien transporté.
    {
        double worst = 0.0;
        for (double vy : {0.0, 0.3, 0.6}) {
            const PrimitiveState P = mkPrim(1.0, 1.0, 0.4, vy);
            const FluidVars f = makeFluidVars(P, eos);
            const ConservedState U = primitiveToConserved(P, eos);
            double F[5]; physicalFlux(f, U, 0, F);
            const double expect = U.S[1] * f.v[0];
            const double sc = std::max(std::fabs(expect), 1e-30);
            worst = std::max(worst, std::fabs(F[2] - expect) / sc);
        }
        report("A3 transverse : F_{S_y} = S_y v^x (pas de p hors diagonale)", worst, 1e-15);
    }

    // A4 — IDENTITÉ DE FLUX INDÉPENDANTE : F_tau = S^x - D v^x doit égaler
    // (tau + p) v^x. Les deux écritures viennent de dérivations différentes
    // (cf. en-tête de Flux.hpp) : leur accord valide la dérivation.
    {
        double worst = 0.0;
        for (double v : {0.05, 0.5, 0.95}) {
            for (double uu : {0.01, 1.0, 100.0}) {
                const PrimitiveState P = mkPrim(1.0, uu, v);
                const FluidVars f = makeFluidVars(P, eos);
                const ConservedState U = primitiveToConserved(P, eos);
                double F[5]; physicalFlux(f, U, 0, F);
                const double alt = U.S[0] - U.D * f.v[0];
                worst = std::max(worst, std::fabs(F[4] - alt) / std::max(std::fabs(alt), 1e-30));
            }
        }
        report("A4 identite F_tau : (tau+p)v == S^x - D v^x", worst, 1e-13);
    }

    // A5 — limite newtonienne : F_{S_x} -> rho v² + p  quand v << 1.
    {
        std::printf("       limite newtonienne du flux de moment :\n");
        double worstRatio = 0.0;
        for (double v : {1e-3, 1e-2, 1e-1}) {
            const PrimitiveState P = mkPrim(1.0, 1e-4, v);
            const FluidVars f = makeFluidVars(P, eos);
            const ConservedState U = primitiveToConserved(P, eos);
            double F[5]; physicalFlux(f, U, 0, F);
            const double p = eos.pressure(P.rho, P.u);
            const double newton = P.rho * v * v + p;
            const double rel = std::fabs(F[1] - newton) / newton;
            std::printf("         v = %-8.0e  F_Sx = %.8e  newtonien = %.8e  ecart/v^2 = %.3f\n",
                        v, F[1], newton, rel / (v * v));
            worstRatio = std::max(worstRatio, rel / (v * v));
        }
        report("A5 limite newtonienne : ecart d'ordre v^2 (rapport borne)", worstRatio, 50.0);
    }
}

// =============================================================================
//  B — VITESSES CARACTÉRISTIQUES
// =============================================================================
void testCharacteristics() {
    const IdealGasEOS eos(4.0 / 3.0);

    // B1 — limite v -> 0 : lambda_pm -> ± c_s.
    {
        double worst = 0.0;
        for (double uu : {0.1, 1.0, 10.0}) {
            const PrimitiveState P = mkPrim(1.0, uu, 0.0);
            const FluidVars f = makeFluidVars(P, eos);
            const SignalSpeeds s = characteristicSpeeds(f, 0);
            const double cs = std::sqrt(f.cs2);
            worst = std::max(worst, std::fabs(s.lambdaPlus - cs) / cs);
            worst = std::max(worst, std::fabs(s.lambdaMinus + cs) / cs);
        }
        report("B1 v->0 : lambda_pm = +/- c_s", worst, 1e-14);
    }

    // B2 — limite c_s -> 0 (fluide froid) : advection pure avec le fluide.
    {
        double worst = 0.0;
        for (double v : {0.0, 0.3, 0.9}) {
            PrimitiveState P = mkPrim(1.0, 0.0, v);   // u = 0 => c_s = 0
            const FluidVars f = makeFluidVars(P, eos);
            const SignalSpeeds s = characteristicSpeeds(f, 0);
            worst = std::max(worst, std::fabs(s.lambdaPlus - v));
            worst = std::max(worst, std::fabs(s.lambdaMinus - v));
            worst = std::max(worst, std::fabs(s.lambda0 - v));
        }
        report("B2 c_s->0 : advection pure (lambda = v)", worst, 1e-15);
    }

    // B3 — IDENTITÉ INDÉPENDANTE : sans vitesse transverse, la formule
    // generale doit se reduire EXACTEMENT a l'addition relativiste des
    // vitesses (v +/- c_s)/(1 +/- v c_s). C'est un oracle algebrique distinct.
    {
        double worst = 0.0;
        for (double v : {-0.95, -0.5, 0.0, 0.3, 0.7, 0.99}) {
            for (double uu : {0.01, 1.0, 50.0}) {
                const PrimitiveState P = mkPrim(1.0, uu, v);
                const FluidVars f = makeFluidVars(P, eos);
                const SignalSpeeds s = characteristicSpeeds(f, 0);
                const double cs = std::sqrt(f.cs2);
                const double refP = (v + cs) / (1.0 + v * cs);
                const double refM = (v - cs) / (1.0 - v * cs);
                worst = std::max(worst, std::fabs(s.lambdaPlus - refP));
                worst = std::max(worst, std::fabs(s.lambdaMinus - refM));
            }
        }
        report("B3 sans transverse == addition relativiste (oracle algebrique)", worst, 1e-13);
    }

    // B4 — CAUSALITÉ : |lambda| < 1 sur un large balayage AVEC transverse.
    // C'est ici que « v +/- c_s » echouerait grossierement.
    {
        double worstAbs = 0.0;
        int n = 0;
        for (double vx = -0.95; vx <= 0.95; vx += 0.05) {
            for (double vt = 0.0; vt <= 0.9; vt += 0.05) {
                if (vx * vx + vt * vt >= 0.999) continue;
                for (double uu : {0.01, 1.0, 100.0}) {
                    const PrimitiveState P = mkPrim(1.0, uu, vx, vt);
                    const FluidVars f = makeFluidVars(P, eos);
                    const SignalSpeeds s = characteristicSpeeds(f, 0);
                    worstAbs = std::max(worstAbs, std::max(std::fabs(s.lambdaPlus),
                                                            std::fabs(s.lambdaMinus)));
                    ++n;
                }
            }
        }
        std::printf("       causalite verifiee sur %d etats (avec vitesse transverse)\n", n);
        reportBool("B4 |lambda| < 1 partout (causalite)", worstAbs < 1.0);

        // Contre-exemple explicite : montre que v+c_s violerait la causalite.
        const PrimitiveState P = mkPrim(1.0, 100.0, 0.9, 0.3);
        const FluidVars f = makeFluidVars(P, eos);
        const SignalSpeeds s = characteristicSpeeds(f, 0);
        const double naive = f.v[0] + std::sqrt(f.cs2);
        std::printf("       contre-exemple (vx=0.9, vt=0.3, c_s=%.4f) :\n"
                    "         naif v+c_s = %.4f  %s   |  correct lambda+ = %.4f\n",
                    std::sqrt(f.cs2), naive,
                    (naive >= 1.0 ? "(SUPERLUMINIQUE !)" : "(sous-luminique mais faux)"),
                    s.lambdaPlus);
    }
}

// =============================================================================
//  C — HLLE ISOLÉ
// =============================================================================
void testHLLE() {
    const IdealGasEOS eos(5.0 / 3.0);

    // C1 — consistance : gauche == droite => flux numerique == flux physique.
    {
        double worst = 0.0;
        for (double v : {-0.8, -0.2, 0.0, 0.4, 0.9}) {
            const PrimitiveState P = mkPrim(1.7, 2.3, v, 0.2);
            RiemannInput in;
            in.fL = in.fR = makeFluidVars(P, eos);
            in.uL = in.uR = primitiveToConserved(P, eos);
            double Fn[5], Fp[5];
            hlleFlux(in, 0, Fn);
            physicalFlux(in.fL, in.uL, 0, Fp);
            for (int k = 0; k < 5; ++k) {
                const double sc = std::max(std::fabs(Fp[k]), 1e-30);
                worst = std::max(worst, std::fabs(Fn[k] - Fp[k]) / sc);
            }
        }
        report("C1 consistance : F_HLLE(U,U) == F_phys(U)", worst, 1e-15);
    }

    // C2 — supersonique : si toutes les ondes vont vers +x, le flux est le
    // flux amont GAUCHE exactement (et symetriquement).
    {
        const PrimitiveState PL = mkPrim(1.0, 0.01, 0.99);
        const PrimitiveState PR = mkPrim(1.2, 0.01, 0.98);
        RiemannInput in;
        in.fL = makeFluidVars(PL, eos); in.fR = makeFluidVars(PR, eos);
        in.uL = primitiveToConserved(PL, eos); in.uR = primitiveToConserved(PR, eos);
        double Fn[5], Fp[5];
        hlleFlux(in, 0, Fn);
        physicalFlux(in.fL, in.uL, 0, Fp);
        double worst = 0.0;
        for (int k = 0; k < 5; ++k)
            worst = std::max(worst, std::fabs(Fn[k] - Fp[k]) / std::max(std::fabs(Fp[k]), 1e-30));
        report("C2 supersonique droite : flux = amont gauche exact", worst, 1e-15);
    }

    // C3 — symetrie : inverser gauche/droite et le signe de v doit changer le
    // signe des flux de masse/energie et conserver celui du flux de moment.
    {
        const PrimitiveState PL = mkPrim(1.0, 1.0,  0.3);
        const PrimitiveState PR = mkPrim(0.5, 0.4, -0.2);
        RiemannInput a, b;
        a.fL = makeFluidVars(PL, eos); a.fR = makeFluidVars(PR, eos);
        a.uL = primitiveToConserved(PL, eos); a.uR = primitiveToConserved(PR, eos);
        // Etat miroir : on echange L<->R et on inverse les vitesses.
        const PrimitiveState mL = mkPrim(0.5, 0.4,  0.2);
        const PrimitiveState mR = mkPrim(1.0, 1.0, -0.3);
        b.fL = makeFluidVars(mL, eos); b.fR = makeFluidVars(mR, eos);
        b.uL = primitiveToConserved(mL, eos); b.uR = primitiveToConserved(mR, eos);
        double Fa[5], Fb[5];
        hlleFlux(a, 0, Fa); hlleFlux(b, 0, Fb);
        double worst = 0.0;
        worst = std::max(worst, std::fabs(Fa[0] + Fb[0]) / std::max(std::fabs(Fa[0]), 1e-30));
        worst = std::max(worst, std::fabs(Fa[1] - Fb[1]) / std::max(std::fabs(Fa[1]), 1e-30));
        worst = std::max(worst, std::fabs(Fa[4] + Fb[4]) / std::max(std::fabs(Fa[4]), 1e-30));
        report("C3 symetrie miroir des flux", worst, 1e-13);
    }

    // C4 — finitude sur des regimes extremes.
    {
        bool allFinite = true;
        for (double v : {0.999, -0.999}) {
            for (double uu : {1e-8, 1e6}) {
                const PrimitiveState PL = mkPrim(1e-4, uu, v);
                const PrimitiveState PR = mkPrim(1e4, uu * 1e-3, -v);
                RiemannInput in;
                in.fL = makeFluidVars(PL, eos); in.fR = makeFluidVars(PR, eos);
                in.uL = primitiveToConserved(PL, eos); in.uR = primitiveToConserved(PR, eos);
                double F[5]; hlleFlux(in, 0, F);
                for (int k = 0; k < 5; ++k) if (!std::isfinite(F[k])) allFinite = false;
            }
        }
        reportBool("C4 flux fini en regimes relativistes extremes", allFinite);
    }
}

// =============================================================================
//  D — CONSERVATION GLOBALE DISCRÈTE
// =============================================================================
// Test conçu pour ÉCHOUER si deux cellules adjacentes n'utilisaient pas
// exactement le même flux de face : sur un domaine périodique, la somme des
// conservées doit se télescoper à l'arrondi près.
void testGlobalConservation() {
    Solver1DConfig cfg;
    cfg.nx = 128; cfg.xmin = 0.0; cfg.xmax = 1.0;
    cfg.cfl = 0.4; cfg.gamma = 5.0 / 3.0;
    cfg.recon = Reconstruction::PLM_MC;
    cfg.bc = BoundaryCondition::Periodic;
    cfg.floors.enabled = true;

    Solver1D s(cfg);
    s.initialize([](double x) {
        // Profil lisse mais fortement non trivial (gradients de rho, u et v).
        const double rho = 1.0 + 0.5 * std::sin(2.0 * M_PI * x);
        const double u   = 1.0 + 0.3 * std::cos(4.0 * M_PI * x);
        const double v   = 0.2 * std::sin(2.0 * M_PI * x);
        return mkPrim(rho, u, v);
    });

    double D0, S0[3], T0;
    s.globalSums(D0, S0, T0);
    const int nsteps = s.advanceTo(0.3);
    double D1, S1[3], T1;
    s.globalSums(D1, S1, T1);

    const double eD = std::fabs(D1 - D0) / std::fabs(D0);
    const double eS = std::fabs(S1[0] - S0[0]) / std::max(std::fabs(S0[0]), 1e-30);
    const double eT = std::fabs(T1 - T0) / std::fabs(T0);
    std::printf("       domaine periodique, %d pas : dSum(D)=%.3e dSum(Sx)=%.3e dSum(tau)=%.3e\n",
                nsteps, eD, eS, eT);
    std::printf("       floors: rho=%lld u=%lld | coldAtMachinePrec=%lld | recoveryFail=%lld\n",
                s.counters().rhoFloored, s.counters().uFloored,
                s.counters().coldAtMachinePrecision, s.counters().recoveryFailures);

    report("D1 conservation de sum(D) (periodique)", eD, 1e-12);
    report("D2 conservation de sum(tau) (periodique)", eT, 1e-12);
    reportBool("D3 aucun floor necessaire sur un probleme lisse",
               s.counters().totalFloors() == 0);
    reportBool("D4 aucun echec de recovery", s.counters().recoveryFailures == 0);
}

// =============================================================================
//  E — CONVERGENCE SUR SOLUTION LISSE (avant les chocs)
// =============================================================================
// Advection d'une perturbation de densite faible sur fond uniforme : la
// solution exacte est le profil initial translate de v*t.
// Leçon de S1 appliquee : on montre les erreurs BRUTES et on ne mesure l'ordre
// que la ou l'erreur est franchement au-dessus du plancher d'arrondi.
void testSmoothConvergence() {
    const double v0 = 0.2, tEnd = 0.5;
    const double amp = 1e-3;   // faible => reste dans le regime lineaire/lisse

    std::printf("       advection d'une perturbation lisse (v=%.2f, t=%.2f, amp=%.0e) :\n",
                v0, tEnd, amp);
    std::printf("         %-6s %-12s %-12s %-12s %s\n", "nx", "L1", "L2", "Linf", "ordre L1");

    double prevL1 = 0.0;
    double bestOrder = 0.0;
    bool measuredAboveFloor = false;

    for (int nx : {50, 100, 200, 400}) {
        Solver1DConfig cfg;
        cfg.nx = nx; cfg.xmin = 0.0; cfg.xmax = 1.0;
        cfg.cfl = 0.4; cfg.gamma = 5.0 / 3.0;
        cfg.recon = Reconstruction::PLM_MC;
        cfg.bc = BoundaryCondition::Periodic;

        auto profile = [&](double x) {
            return 1.0 + amp * std::sin(2.0 * M_PI * x);
        };
        Solver1D s(cfg);
        s.initialize([&](double x) { return mkPrim(profile(x), 1.0, v0); });
        s.advanceTo(tEnd);

        double l1 = 0.0, l2 = 0.0, linf = 0.0;
        for (int i = 0; i < nx; ++i) {
            const double x = s.xc(Solver1D::NG + i);
            double xs = x - v0 * tEnd;
            xs -= std::floor(xs);                     // periodicite
            const double exact = profile(xs);
            const double e = std::fabs(s.primitives()[Solver1D::NG + i].rho - exact);
            l1 += e; l2 += e * e; linf = std::max(linf, e);
        }
        l1 /= nx; l2 = std::sqrt(l2 / nx);

        double order = 0.0;
        if (prevL1 > 0.0) order = std::log2(prevL1 / l1);
        std::printf("         %-6d %-12.4e %-12.4e %-12.4e %s%.2f\n",
                    nx, l1, l2, linf, prevL1 > 0.0 ? "" : "   -  ", order);
        // Plancher d'arrondi estime : amp * eps * (nb d'operations) ~ 1e-3*1e-13
        if (prevL1 > 1e-13 && l1 > 1e-13) {
            bestOrder = std::max(bestOrder, order);
            measuredAboveFloor = true;
        }
        prevL1 = l1;
    }
    reportBool("E1 mesure effectuee au-dessus du plancher d'arrondi", measuredAboveFloor);
    report("E2 ordre de convergence PLM+SSPRK2 >= 1.7", bestOrder >= 1.7 ? 0.0 : 1.0, 1e-15);
    std::printf("       -> ordre max observe : %.2f (attendu ~2 pour PLM/MC sur solution lisse ;\n"
                "          le limiteur ecrete les extrema, ce qui degrade legerement l'ordre)\n",
                bestOrder);
}

// =============================================================================
//  F — TUBES À CHOC vs ORACLE DE RIEMANN EXACT
// =============================================================================

// F0 — l'ORACLE lui-meme doit etre verifie avant de servir de reference :
// les relations de Rankine-Hugoniot doivent tenir a travers le choc calcule.
void testExactSolverJumpConditions() {
    const double g = 5.0 / 3.0;
    std::printf("       verification des relations de saut de l'oracle (RH) :\n");
    double worst = 0.0;
    for (double pPost : {2.0, 10.0, 1000.0}) {
        RiemannState pre; pre.rho = 1.0; pre.p = 1.0; pre.v = 0.0;
        const ShockResult s = shockBranch(pre, pPost, g, +1);
        if (!s.ok) { worst = 1.0; continue; }

        // ÉVALUATION EN LONG DOUBLE — mais sur les valeurs (Vs, vPost, rhoPost)
        // calculées en double par l'oracle. Ce choix est délibéré et permet de
        // DISCRIMINER deux causes possibles d'un résidu non nul :
        //   * si le résidu reste grand en long double  -> l'oracle est faux ;
        //   * s'il s'effondre                          -> c'est l'arithmétique
        //     du TEST qui était mal conditionnée (les flux relatifs au choc
        //     font intervenir (v - Vs) et W², fortement amplifiés quand
        //     Vs -> 1), pas la solution.
        // Les vitesses proches de 1 rendent 1-Vs² catastrophiquement sensible :
        // à Vs = 0.999393, une erreur relative eps sur Vs devient ~1.6e-13 sur
        // (1-Vs), puis est encore amplifiée par W² ~ 800 dans les flux.
        using LD = long double;
        const LD gl = static_cast<LD>(g);
        auto W = [](LD v) { return 1.0L / sqrtl(1.0L - v * v); };
        auto hOf = [&](LD rho, LD p) { return 1.0L + gl / (gl - 1.0L) * p / rho; };
        const LD Vs = static_cast<LD>(s.shockSpeed);
        const LD r1 = static_cast<LD>(pre.rho), p1 = static_cast<LD>(pre.p), v1 = static_cast<LD>(pre.v);
        const LD r2 = static_cast<LD>(s.rhoPost), p2 = static_cast<LD>(pPost), v2 = static_cast<LD>(s.vPost);

        const LD m1 = r1 * W(v1) * (v1 - Vs);
        const LD m2 = r2 * W(v2) * (v2 - Vs);
        const LD mom1 = r1 * hOf(r1, p1) * W(v1) * W(v1) * v1 * (v1 - Vs) + p1;
        const LD mom2 = r2 * hOf(r2, p2) * W(v2) * W(v2) * v2 * (v2 - Vs) + p2;
        const LD en1 = r1 * hOf(r1, p1) * W(v1) * W(v1) * (v1 - Vs) + p1 * Vs;
        const LD en2 = r2 * hOf(r2, p2) * W(v2) * W(v2) * (v2 - Vs) + p2 * Vs;

        // NORMALISATION — par l'échelle des TERMES SOMMÉS, pas par le résultat.
        // ⚠️ Même piège qu'en S1-A3, et je l'ai reproduit : la première version
        // divisait par |mom1|. Or au choc fort, mom2 = ρhW²v(v-Vs) + p est la
        // somme de deux termes d'ordre ~900 et ~1000 qui se compensent pour
        // valoir ~1 : trois chiffres significatifs sont perdus par annulation,
        // et diviser par ce résultat minuscule amplifie l'arrondi d'autant.
        // On normalise donc par max des termes réellement additionnés, qui est
        // la seule échelle qui ne s'effondre pas là où l'on mesure.
        const LD momScale = fmaxl(fabsl(r2 * hOf(r2, p2) * W(v2) * W(v2) * v2 * (v2 - Vs)),
                                   fmaxl(fabsl(p2), fabsl(mom1)));
        const LD enScale  = fmaxl(fabsl(r2 * hOf(r2, p2) * W(v2) * W(v2) * (v2 - Vs)),
                                   fmaxl(fabsl(p2 * Vs), fabsl(en1)));
        const LD mScale   = fmaxl(fabsl(m1), fabsl(m2));

        const double eM = static_cast<double>(fabsl(m1 - m2) / fmaxl(mScale, 1e-30L));
        const double eP = static_cast<double>(fabsl(mom1 - mom2) / fmaxl(momScale, 1e-30L));
        const double eE = static_cast<double>(fabsl(en1 - en2) / fmaxl(enScale, 1e-30L));
        const double Ws2 = 1.0 / (1.0 - s.shockSpeed * s.shockSpeed);
        std::printf("         p_post=%-8.0f Vs=%.6f Ws^2=%.3e  RH: masse %.2e  moment %.2e  energie %.2e\n",
                    pPost, s.shockSpeed, Ws2, eM, eP, eE);
        worst = std::max(worst, std::max(eM, std::max(eP, eE)));
    }
    report("F0 oracle : relations de Rankine-Hugoniot verifiees", worst, 1e-10);
}

struct ShockTubeResult {
    double l1rho, l1p, l1v;
    long long floors, coldFlags, recoveryFails;
    int steps;
};

ShockTubeResult runShockTube(const char* label,
                             const RiemannState& L, const RiemannState& R,
                             double gamma, double tEnd, int nx,
                             Reconstruction recon, double cfl, bool verbose) {
    Solver1DConfig cfg;
    cfg.nx = nx; cfg.xmin = -0.5; cfg.xmax = 0.5;
    cfg.cfl = cfl; cfg.gamma = gamma;
    cfg.recon = recon;
    cfg.bc = BoundaryCondition::Outflow;

    const IdealGasEOS eos(gamma);
    Solver1D s(cfg);
    s.initialize([&](double x) {
        const RiemannState& K = (x < 0.0) ? L : R;
        return mkPrim(K.rho, eos.internalEnergy(K.rho, K.p), K.v);
    });
    const int steps = s.advanceTo(tEnd);

    const ExactRiemannSolution sol = solveExactRiemannSRHD(L, R, gamma);

    ShockTubeResult res{};
    res.steps = steps;
    double l1r = 0.0, l1p = 0.0, l1v = 0.0;
    for (int i = 0; i < nx; ++i) {
        const double x = s.xc(Solver1D::NG + i);
        const double xi = x / tEnd;
        const RiemannState ex = sampleExactRiemannSRHD(sol, L, R, gamma, xi);
        const PrimitiveState& P = s.primitives()[Solver1D::NG + i];
        const double W = lorentzFactor(P);
        const double vnum = P.utilde[0] / W;
        const double pnum = eos.pressure(P.rho, P.u);
        l1r += std::fabs(P.rho - ex.rho);
        l1p += std::fabs(pnum - ex.p);
        l1v += std::fabs(vnum - ex.v);
    }
    res.l1rho = l1r / nx; res.l1p = l1p / nx; res.l1v = l1v / nx;
    res.floors = s.counters().totalFloors();
    res.coldFlags = s.counters().coldAtMachinePrecision;
    res.recoveryFails = s.counters().recoveryFailures;

    if (verbose) {
        std::printf("         %-22s nx=%-4d t=%.2f CFL=%.2f | L1: rho %.3e  p %.3e  v %.3e\n",
                    label, nx, tEnd, cfl, res.l1rho, res.l1p, res.l1v);
        std::printf("           p*=%.6e v*=%.6e | pas=%d | floors=%lld coldFlag=%lld recFail=%lld\n",
                    sol.pStar, sol.vStar, steps, res.floors, res.coldFlags, res.recoveryFails);
    }
    return res;
}

void testShockTubes() {
    // Problemes classiques SRHD (famille Marti & Muller). Conditions initiales,
    // temps, resolution et CFL explicitement consignes.
    struct TubeCfg { const char* name; RiemannState L, R; double g, tEnd; };
    const TubeCfg tubes[] = {
        // Probleme 1 : choc relativiste modere.
        {"MM1 choc modere",  {10.0, 13.33, 0.0}, {1.0, 1e-6, 0.0}, 5.0/3.0, 0.35},
        // Probleme 2 : blast wave relativiste fort (p_L/p_R = 1e5).
        {"MM2 blast fort",   {1.0, 1000.0, 0.0}, {1.0, 0.01,  0.0}, 5.0/3.0, 0.35},
        // Probleme avec vitesse initiale non nulle (contact + choc).
        {"MM3 avec v initial",{1.0, 1.0, 0.5},   {1.0, 10.0, 0.0}, 4.0/3.0, 0.30},
    };

    std::printf("       tubes a choc, comparaison a l'oracle EXACT :\n");
    bool allOk = true;
    long long totalFloors = 0, totalRecFail = 0;

    for (const TubeCfg& t : tubes) {
        // Etude de convergence en resolution : l'erreur L1 doit decroitre.
        double prevL1 = 0.0;
        bool decreasing = true;
        for (int nx : {200, 400, 800}) {
            const ShockTubeResult r = runShockTube(t.name, t.L, t.R, t.g, t.tEnd, nx,
                                                   Reconstruction::PLM_MC, 0.4, nx == 800);
            totalFloors += r.floors;
            totalRecFail += r.recoveryFails;
            if (prevL1 > 0.0 && r.l1rho > prevL1) decreasing = false;
            prevL1 = r.l1rho;
            if (!std::isfinite(r.l1rho)) allOk = false;
        }
        if (!decreasing) allOk = false;
        std::printf("           -> L1(rho) decroissante avec la resolution : %s\n",
                    decreasing ? "OUI" : "NON");
    }
    reportBool("F1 tubes a choc : erreurs finies et decroissantes", allOk);
    reportBool("F2 aucun echec de recovery sur les tubes", totalRecFail == 0);
    std::printf("       -> floors totaux sur tous les tubes/resolutions : %lld\n", totalFloors);

    // F3 — le mode PREMIER ORDRE doit fonctionner : c'est l'outil de
    // diagnostic qui distingue « bug du solveur de Riemann » de « oscillation
    // de la reconstruction d'ordre 2 ».
    const TubeCfg& t = tubes[0];
    const ShockTubeResult pcm = runShockTube("PCM (1er ordre)", t.L, t.R, t.g, t.tEnd, 400,
                                             Reconstruction::PiecewiseConstant, 0.4, true);
    const ShockTubeResult plm = runShockTube("PLM/MC (2e ordre)", t.L, t.R, t.g, t.tEnd, 400,
                                             Reconstruction::PLM_MC, 0.4, true);
    reportBool("F3 mode 1er ordre disponible et stable",
               std::isfinite(pcm.l1rho) && pcm.recoveryFails == 0);
    reportBool("F4 PLM plus precis que PCM (L1 rho)", plm.l1rho < pcm.l1rho);
    std::printf("       -> L1(rho) : PCM %.4e  vs  PLM %.4e  (gain x%.2f)\n",
                pcm.l1rho, plm.l1rho, pcm.l1rho / std::max(plm.l1rho, 1e-30));
}

// =============================================================================
//  G — INVARIANCE PAR BOOST DE LORENTZ
// =============================================================================
// Objectif : detecter une dependance CACHEE au referentiel dans les relations
// primitives/conservees/flux.
//
// ⚠️ Une comparaison cellule-a-cellule a temps de coordonnee identique n'a PAS
// de sens relativiste (la simultaneite n'est pas invariante, et le domaine
// boosté ne correspond pas au meme evenement). On teste donc ce qui EST
// invariant : les SCALAIRES du fluide.
//
// Methode : on prend un etat, on le boost, et on verifie que les scalaires
// reconstruits (rho, u, p, et la norme u^mu u_mu = -1) sont identiques. Cela
// teste exactement la chaine primitives -> conservees -> primitives sous
// changement de referentiel, qui est ce qui pourrait cacher un biais.
void testLorentzBoostInvariance() {
    const IdealGasEOS eos(4.0 / 3.0);
    std::printf("       invariance des scalaires du fluide sous boost :\n");
    double worstRho = 0.0, worstU = 0.0, worstNorm = 0.0;

    for (double vb : {0.0, 0.3, 0.7, 0.95, 0.99}) {
        // Etat au repos dans son referentiel propre.
        const double rho0 = 2.5, u0 = 1.7;
        // Boost : dans le nouveau referentiel le fluide a la vitesse vb.
        const PrimitiveState P = mkPrim(rho0, u0, vb);
        const ConservedState U = primitiveToConserved(P, eos);

        PrimitiveState back; RecoveryStatus st;
        const bool ok = conservedToPrimitive(U, eos, back, st);
        if (!ok) { worstRho = 1.0; continue; }

        // rho et u sont des SCALAIRES : invariants par construction, ils
        // doivent etre retrouves quelle que soit la vitesse du referentiel.
        worstRho = std::max(worstRho, std::fabs(back.rho - rho0) / rho0);
        worstU   = std::max(worstU,   std::fabs(back.u - u0) / u0);

        // Normalisation de la quadrivitesse : u^mu u_mu = -1 exactement.
        const double W = lorentzFactor(back);
        const double norm = -W * W + back.utilde[0] * back.utilde[0]
                                    + back.utilde[1] * back.utilde[1]
                                    + back.utilde[2] * back.utilde[2];
        worstNorm = std::max(worstNorm, std::fabs(norm + 1.0));

        std::printf("         v_boost=%.2f  W=%.4f  err(rho)=%.2e err(u)=%.2e |u.u+1|=%.2e\n",
                    vb, W, std::fabs(back.rho - rho0) / rho0,
                    std::fabs(back.u - u0) / u0, std::fabs(norm + 1.0));
    }
    report("G1 rho invariant sous boost (scalaire)", worstRho, 1e-12);
    report("G2 u invariant sous boost (scalaire)", worstU, 1e-10);
    report("G3 normalisation u^mu u_mu = -1", worstNorm, 1e-13);

    // G4 — invariance du PROBLEME PHYSIQUE : un tube a choc et le meme tube vu
    // depuis un referentiel boosté doivent avoir des solutions exactes reliees
    // par la transformation de vitesses. On compare les SOLUTIONS EXACTES
    // (oracle), ce qui teste la coherence relativiste de notre formulation
    // sans exiger de correspondance cellule-a-cellule.
    {
        const double g = 5.0 / 3.0, vb = 0.4;
        RiemannState L{10.0, 13.33, 0.0}, R{1.0, 1e-6, 0.0};
        auto boost = [&](double v) { return (v + vb) / (1.0 + v * vb); };
        RiemannState Lb = L, Rb = R;
        Lb.v = boost(L.v); Rb.v = boost(R.v);

        const ExactRiemannSolution s0 = solveExactRiemannSRHD(L, R, g);
        const ExactRiemannSolution sb = solveExactRiemannSRHD(Lb, Rb, g);

        // La pression du contact est un SCALAIRE : identique dans les deux
        // referentiels. La vitesse du contact se transforme par addition
        // relativiste.
        const double ep = std::fabs(sb.pStar - s0.pStar) / s0.pStar;
        const double evExpected = boost(s0.vStar);
        const double ev = std::fabs(sb.vStar - evExpected);
        std::printf("         boost du probleme (vb=%.2f) : p* invariant a %.2e,\n"
                    "           v* suit l'addition relativiste a %.2e\n", vb, ep, ev);
        report("G4a p* invariant sous boost du probleme", ep, 1e-9);
        report("G4b v* suit l'addition relativiste", ev, 1e-9);
    }
}

// =============================================================================
//  H — HDF5
// =============================================================================
void testHDF5() {
    Solver1DConfig cfg;
    cfg.nx = 64; cfg.xmin = 0.0; cfg.xmax = 1.0;
    cfg.gamma = 5.0 / 3.0;
    cfg.bc = BoundaryCondition::Periodic;
    Solver1D s(cfg);
    s.initialize([](double x) { return mkPrim(1.0 + 0.2 * std::sin(2.0 * M_PI * x), 1.0, 0.1); });
    s.advanceTo(0.05);

    const std::string path = "grmhd_s3_snapshot.h5";
    const bool wrote = writeSnapshotHDF5(s, path);
    reportBool("H1 ecriture HDF5", wrote);
    if (!wrote) return;

    // Un writer non relu n'est pas un writer teste.
    std::vector<double> x, rho, uu, u1;
    double t = -1.0; int nx = -1;
    const bool read = readSnapshotHDF5(path, x, rho, uu, u1, t, nx);
    reportBool("H2 relecture HDF5", read);
    if (!read) return;

    reportBool("H3 dimensions coherentes",
               nx == cfg.nx && (int)rho.size() == cfg.nx && (int)x.size() == cfg.nx);

    double worst = 0.0;
    for (int i = 0; i < cfg.nx; ++i) {
        const PrimitiveState& P = s.primitives()[Solver1D::NG + i];
        worst = std::max(worst, std::fabs(rho[i] - P.rho) / P.rho);
        worst = std::max(worst, std::fabs(u1[i] - P.utilde[0])
                                 / std::max(std::fabs(P.utilde[0]), 1e-30));
    }
    report("H4 aller-retour des donnees exact", worst, 1e-15);
    report("H5 temps correctement serialise", std::fabs(t - s.time()) / s.time(), 1e-15);
    std::printf("       -> fichier '%s' ecrit (noms iharm : prims/RHO, UU, U1..U3)\n"
                "          champs ABSENTS et documentes comme tels : B1..B3 (pas de\n"
                "          champ magnetique avant S5), metrique/FMKS/spin (espace plat).\n",
                path.c_str());
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf("  S3 : solveur SRHD 1D en espace plat (Minkowski)\n");
    std::printf("=====================================================================\n");

    std::printf("\n  A -- flux physiques (avant toute reconstruction)\n");
    testPhysicalFlux();
    std::printf("\n  B -- vitesses caracteristiques relativistes\n");
    testCharacteristics();
    std::printf("\n  C -- solveur de Riemann HLLE isole\n");
    testHLLE();
    std::printf("\n  D -- conservation globale discrete\n");
    testGlobalConservation();
    std::printf("\n  E -- convergence sur solution LISSE\n");
    testSmoothConvergence();
    std::printf("\n  F -- tubes a choc vs oracle de Riemann EXACT\n");
    testExactSolverJumpConditions();
    testShockTubes();
    std::printf("\n  G -- invariance par boost de Lorentz\n");
    testLorentzBoostInvariance();
    std::printf("\n  H -- serialisation HDF5\n");
    testHDF5();

    std::printf("---------------------------------------------------------------------\n");
    std::printf("  %d tests executes.\n", g_tests);
    if (g_failures == 0)
        std::printf("  S3 VALIDE -- solveur SRHD de reference etabli.\n");
    else
        std::printf("  %d ECHEC(S) -- S3 NON valide.\n", g_failures);
    std::printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
