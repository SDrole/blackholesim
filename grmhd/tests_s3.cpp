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
#include <limits>
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
        report("A3 transverse : F_{S_y} = S_y v^x (no pressure term off diagonal)", worst, 1e-15);
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
        std::printf("       causality verified on %d states (including transverse velocity)\n", n);
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
        report("C2 right-going supersonic: flux = exact left upwind flux", worst, 1e-15);
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
    std::printf("       periodic domain, %d steps : dSum(D)=%.3e dSum(Sx)=%.3e dSum(tau)=%.3e\n",
                nsteps, eD, eS, eT);
    std::printf("       floors: rho=%lld u=%lld | coldAtMachinePrec=%lld | recoveryFail=%lld\n",
                s.counters().rhoFloored, s.counters().uFloored,
                s.counters().coldAtMachinePrecision, s.counters().recoveryFailures);

    report("D1 conservation de sum(D) (periodique)", eD, 1e-12);
    report("D2 conservation de sum(tau) (periodique)", eT, 1e-12);
    reportBool("D3 no floor needed on a smooth problem",
               s.counters().totalFloors() == 0);
    reportBool("D4 no primitive-recovery failure", s.counters().recoveryFailures == 0);
    reportBool("D5 no reconstruction fallback (instrumented)",
               s.counters().reconstructionFallbacks == 0);
    reportBool("D6 no U<-P resynchronization (scheme remains conservative)",
               s.counters().conservativeResyncs == 0);
    std::printf("       reconstructionFallbacks=%lld  conservativeResyncs=%lld\n",
                s.counters().reconstructionFallbacks, s.counters().conservativeResyncs);
}

// D7 — CONTRAT DE CONSERVATION : un recovery normal ne doit PAS toucher U.
// Test discriminant : on compare U bit a bit avant/apres un appel a la
// recuperation des primitives. Toute resynchronisation P->U injecterait
// l'erreur d'arrondi de l'aller-retour et ferait echouer ce test.
void testConservationContract() {
    Solver1DConfig cfg;
    cfg.nx = 64; cfg.xmin = 0.0; cfg.xmax = 1.0;
    cfg.cfl = 0.4; cfg.gamma = 5.0 / 3.0;
    cfg.recon = Reconstruction::PLM_MC;
    cfg.bc = BoundaryCondition::Periodic;

    Solver1D s(cfg);
    s.initialize([](double x) {
        return mkPrim(1.0 + 0.4 * std::sin(2.0 * M_PI * x),
                      1.0 + 0.2 * std::cos(2.0 * M_PI * x),
                      0.25 * std::sin(2.0 * M_PI * x));
    });

    s.step(0.2 * s.dx());
    s.step(0.2 * s.dx());

    // Verification bit-a-bit du CONTRAT : les cellules ou aucune correction
    // n'a eu lieu ne doivent pas avoir subi d'aller-retour parasite. On le
    // teste via le compteur : zero resynchronisation => U intact par
    // construction.
    reportBool("D7a no resynchronization on a healthy problem",
               s.counters().conservativeResyncs == 0);

    // Comparaison directe : on refait un recovery sur un etat GELE et on
    // verifie que U ressort IDENTIQUE bit a bit.
    Solver1D t(cfg);
    t.initialize([](double x) {
        return mkPrim(1.0 + 0.4 * std::sin(2.0 * M_PI * x), 1.0, 0.1);
    });
    const std::vector<ConservedState> u0 = t.conserved();
    t.step(0.2 * t.dx());
    const std::vector<ConservedState> u1 = t.conserved();
    // Reappliquer un pas de duree nulle ne doit rien changer du tout.
    t.step(0.0);
    const std::vector<ConservedState>& u2 = t.conserved();
    bool bitIdentical = true;
    for (int i = Solver1D::NG; i < Solver1D::NG + cfg.nx; ++i) {
        if (u2[i].D != u1[i].D || u2[i].tau != u1[i].tau ||
            u2[i].S[0] != u1[i].S[0]) bitIdentical = false;
    }
    (void)u0;
    reportBool("D7b zero-duration step leaves U bit-identical", bitIdentical);
    reportBool("D7c no recovery failure (Abort policy active)", t.ok());
}

// =============================================================================
//  D8/D9 — CHEMINS D'ÉCHEC DE RECOVERY, RÉELLEMENT FORCÉS
// =============================================================================
// Les tests précédents ont 0 échec de recovery : ils ne peuvent donc RIEN
// prouver sur cette branche. On injecte ici un état conservé inadmissible
// (|S| > E, rejeté par S2 avec EnergyBelowMomentum) pour exercer le contrat.
//
// DISCRIMINANCE — vérifiée empiriquement, pas postulée.
// L'ancien comportement (pas de rollback ; `t_ += dt` malgré l'échec) a été
// réintroduit temporairement et la suite relancée. Résultat mesuré :
//   * D8c et D8g ÉCHOUENT  -> ils testent bien l'avance du temps ;
//   * D8d PASSAIT QUAND MÊME -> il ne testait RIEN.
// Raison : un état conservé inadmissible injecté avant le pas est rejeté dès
// l'étage 1, donc AVANT que `cons_` n'ait été touché ; « l'état est intact »
// est alors vrai par accident, rollback ou pas. Le rollback n'a d'effet
// observable qu'à l'étage 2, quand `cons_` porte déjà la mise à jour de
// l'étage 1 — état qu'on ne peut pas atteindre par injection d'état puisque
// l'étage 1 le rejetterait. D'où D8h, qui injecte une FAUTE (et non un état)
// à l'étage 2. D8d est conservé comme couverture du cas « échec précoce »,
// mais son libellé dit explicitement qu'il ne teste pas le rollback.
//
// Mesures avec l'ancien comportement réintroduit :
//   D8c, D8g          -> FAIL
//   D8h2, D8h3, D8h4  -> FAIL, avec max|dU| = 7.889e-04, max|dP| = 7.850e-04
// Avec le code courant : max|dU| = max|dP| = 0.000e+00 (égalité bit à bit).

ConservedState inadmissibleState() {
    // E = tau + D = 2, |S| = 100 >> E : aucune vitesse < c ne peut produire
    // cet état. S2 le rejette pour EnergyBelowMomentum.
    ConservedState U;
    U.D = 1.0; U.tau = 1.0;
    U.S[0] = 100.0; U.S[1] = 0.0; U.S[2] = 0.0;
    return U;
}

Solver1DConfig failCfg(RecoveryFailurePolicy pol) {
    Solver1DConfig cfg;
    cfg.nx = 32; cfg.xmin = 0.0; cfg.xmax = 1.0;
    cfg.cfl = 0.4; cfg.gamma = 5.0 / 3.0;
    cfg.recon = Reconstruction::PLM_MC;
    cfg.bc = BoundaryCondition::Periodic;
    cfg.recoveryPolicy = pol;
    return cfg;
}

void testForcedRecoveryFailureAbort() {
    Solver1D s(failCfg(RecoveryFailurePolicy::Abort));
    s.initialize([](double x) { return mkPrim(1.0 + 0.2 * std::sin(2.0 * M_PI * x), 1.0, 0.1); });

    // Un pas sain d'abord, pour partir d'un état non trivial.
    const double dt = 0.2 * s.dx();
    const StepResult first = s.step(dt);
    reportBool("D8a healthy step succeeds before fault injection", first.ok && s.ok());

    const double tBefore = s.time();
    const int victim = Solver1D::NG + 7;
    s.injectConservedForTest(victim, inadmissibleState());

    // Photographie de TOUT l'état juste après injection.
    const std::vector<ConservedState> snapshot = s.conserved();
    const long long recFailBefore = s.counters().recoveryFailures;

    const StepResult r = s.step(dt);

    reportBool("D8b failed step is reported", !r.ok && !s.ok());
    reportBool("D8c time did NOT advance",
               s.time() == tBefore);

    // ROLLBACK : l'état entier doit être identique à l'instantané, y compris
    // les cellules SAINES. Avec l'ancien code, le premier étage SSPRK2 les
    // avait déjà modifiées -> ce test échouait.
    bool identical = true;
    const std::vector<ConservedState>& now = s.conserved();
    for (size_t i = 0; i < snapshot.size(); ++i) {
        if (now[i].D != snapshot[i].D || now[i].tau != snapshot[i].tau ||
            now[i].S[0] != snapshot[i].S[0]) { identical = false; break; }
    }
    // ATTENTION : ce test n'est PAS discriminant (cf. en-tête de section) —
    // l'échec survient à l'étage 1, avant toute mutation. Le rollback est testé
    // par D8h.
    reportBool("D8d state intact (early stage-1 failure -- NON-discriminating)", identical);
    // Le compteur « evolution acceptee » est RESTAURE par le rollback ; c'est la
    // telemetrie des tentatives rejetees qui porte l'echec. Les deux
    // significations sont disjointes.
    reportBool("D8e failure is counted in rejected-attempt telemetry",
               s.rejectedTelemetry().during.recoveryFailures > 0
               && s.rejectedTelemetry().rejectedSteps == 1);
    reportBool("D8e2 compteurs 'acceptes' restaures par le rollback",
               s.counters().recoveryFailures == recFailBefore);
    reportBool("D8f no non-conservative resynchronization under Abort",
               s.counters().conservativeResyncs == 0);
    std::printf("       Abort : failure='%s' cell=%d stage=%d | t unchanged=%.17g\n",
                toString(r.failure), r.failedCell, r.failedStage, s.time());

    // advanceTo ne doit pas non plus « avaler » l'échec.
    const double tBefore2 = s.time();
    const int nsteps = s.advanceTo(s.time() + 10.0 * dt);
    reportBool("D8g advanceTo stops immediately and does not advance time",
               nsteps == 0 && s.time() == tBefore2);
}

// D8h — LE test du rollback. La faute est injectée à l'ÉTAGE 2, donc après que
// l'étage 1 a déjà écrit dans TOUTES les cellules physiques. Sans rollback,
// l'état observé après l'échec est l'état intermédiaire d'Euler — visiblement
// différent de l'état initial.
void testAbortRollbackAtStage2() {
    Solver1D s(failCfg(RecoveryFailurePolicy::Abort));
    s.initialize([](double x) { return mkPrim(1.0 + 0.2 * std::sin(2.0 * M_PI * x), 1.0, 0.1); });

    const double dt = 0.2 * s.dx();
    s.step(dt);   // un pas sain

    const double tBefore = s.time();
    const std::vector<ConservedState> before = s.conserved();
    const std::vector<PrimitiveState> beforeP = s.primitives();

    // Contrôle préalable : sans la faute, ce pas modifierait réellement l'état.
    // Sans ce contrôle, « l'état est identique » pourrait être trivialement
    // vrai parce que rien n'évolue.
    {
        Solver1D ref(failCfg(RecoveryFailurePolicy::Abort));
        ref.initialize([](double x) { return mkPrim(1.0 + 0.2 * std::sin(2.0 * M_PI * x), 1.0, 0.1); });
        ref.step(dt);
        const std::vector<ConservedState> a = ref.conserved();
        ref.step(dt);
        const std::vector<ConservedState> b = ref.conserved();
        double maxDiff = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            maxDiff = std::max(maxDiff, std::fabs(a[i].tau - b[i].tau));
        reportBool("D8h0 control: a healthy step actually changes the state", maxDiff > 1e-12);
    }

    s.injectRecoveryFaultForTest(2, Solver1D::NG + 11);
    const StepResult r = s.step(dt);

    reportBool("D8h1 failure is correctly detected at STAGE 2",
               !r.ok && r.failedStage == 2 && r.failedCell == Solver1D::NG + 11);
    reportBool("D8h2 time did not advance", s.time() == tBefore);

    // Comparaison bit à bit : le rollback restaure l'instantané exact, il ne
    // « recalcule » pas un état voisin.
    double maxDU = 0.0, maxDP = 0.0;
    const std::vector<ConservedState>& nowU = s.conserved();
    const std::vector<PrimitiveState>& nowP = s.primitives();
    for (size_t i = 0; i < before.size(); ++i) {
        maxDU = std::max(maxDU, std::fabs(nowU[i].D   - before[i].D));
        maxDU = std::max(maxDU, std::fabs(nowU[i].tau - before[i].tau));
        maxDU = std::max(maxDU, std::fabs(nowU[i].S[0] - before[i].S[0]));
        maxDP = std::max(maxDP, std::fabs(nowP[i].rho - beforeP[i].rho));
        maxDP = std::max(maxDP, std::fabs(nowP[i].u   - beforeP[i].u));
    }
    reportBool("D8h3 U restaure a l'identique (bit a bit)", maxDU == 0.0);
    reportBool("D8h4 P restaure a l'identique (bit a bit)", maxDP == 0.0);
    std::printf("       Stage-2 rollback : max|dU|=%.3e  max|dP|=%.3e  t inchange=%.17g\n",
                maxDU, maxDP, s.time());
}

// D11 — TRANSACTIONALITE DES DIAGNOSTICS.
// Scenario exige par la revue : un evenement diagnostic est produit a l'ETAGE 1
// (des floors qui se declenchent reellement), PUIS une faute est injectee a
// l'ETAGE 2. Apres rollback, l'etat ET les diagnostics « evolution acceptee »
// doivent correspondre exactement au contrat documente ; l'evenement de l'etage
// 1 ne doit pas avoir survecu dans `counters()`, mais ne doit pas non plus
// avoir ete PERDU -- il appartient a `rejectedTelemetry()`.
void testDiagnosticsAreTransactional() {
    Solver1DConfig cfg = failCfg(RecoveryFailurePolicy::Abort);
    // Plancher volontairement HAUT : garantit que des floors se declenchent
    // a chaque etage. Sans cela le test serait vide de sens.
    cfg.floors.enabled = true;
    cfg.floors.rhoMin  = 0.9;    // densite initiale ~1 +- 0.2 -> floors surs
    cfg.floors.uMin    = 1e-15;

    auto ic = [](double x) { return mkPrim(1.0 + 0.2 * std::sin(2.0 * M_PI * x), 1.0, 0.1); };

    // Cellule de FAIBLE DENSITE injectee expressement pour que des floors se
    // declenchent a l'ETAGE 1 du pas qui sera ensuite abandonne.
    // (Etat admissible : E = tau + D > |S| = 0, la recovery reussit ; c'est le
    // FLOOR qui doit agir, pas un echec.)
    auto lowDensityCell = []() {
        ConservedState U;
        U.D = 1e-6; U.tau = 1e-8;
        U.S[0] = U.S[1] = U.S[2] = 0.0;
        return U;
    };
    const int floorCell = Solver1D::NG + 4;
    const int faultCell = Solver1D::NG + 9;

    // CONTRE-CONTROLE, sur un solveur separe : la MEME injection, SANS faute,
    // doit reellement produire des floors. Sans cela, « aucun floor en
    // telemetrie » serait vrai trivialement et le test ne prouverait rien.
    {
        Solver1D ctrl(cfg);
        ctrl.initialize(ic);
        ctrl.step(0.2 * ctrl.dx());
        const long long before = ctrl.counters().totalFloors();
        ctrl.injectConservedForTest(floorCell, lowDensityCell());
        const StepResult cr = ctrl.step(0.2 * ctrl.dx());
        reportBool("D11a control: injection really produces floors at stage 1",
                   cr.ok && ctrl.counters().totalFloors() > before);
    }

    Solver1D s(cfg);
    s.initialize(ic);
    s.step(0.2 * s.dx());

    const double dt = 0.2 * s.dx();

    // L'injection doit precede l'INSTANTANE : c'est l'etat au debut du pas qui
    // fait foi, et c'est lui que le rollback doit restaurer.
    s.injectConservedForTest(floorCell, lowDensityCell());

    const double tBefore = s.time();
    const std::vector<ConservedState> u0 = s.conserved();
    const std::vector<PrimitiveState> p0 = s.primitives();
    const FloorCounters c0 = s.counters();
    const long long rejected0 = s.rejectedTelemetry().rejectedSteps;

    // L'etage 1 se deroulera ENTIEREMENT (floors compris) avant que la faute de
    // l'etage 2 n'abandonne le pas.
    s.injectRecoveryFaultForTest(2, faultCell);
    const StepResult r = s.step(dt);

    reportBool("D11b failure correctly reported at stage 2",
               !r.ok && r.failedStage == 2 && r.failedCell == faultCell);
    reportBool("D11c time did not advance", s.time() == tBefore);

    // Etat : bit a bit.
    double maxDU = 0.0;
    const std::vector<ConservedState>& nowU = s.conserved();
    const std::vector<PrimitiveState>& nowP = s.primitives();
    for (size_t i = 0; i < u0.size(); ++i) {
        maxDU = std::max(maxDU, std::fabs(nowU[i].D   - u0[i].D));
        maxDU = std::max(maxDU, std::fabs(nowU[i].tau - u0[i].tau));
        maxDU = std::max(maxDU, std::fabs(nowP[i].rho - p0[i].rho));
    }
    reportBool("D11d etat restaure a l'identique", maxDU == 0.0);

    // Diagnostics « evolution acceptee » : restaures a l'identique, y compris
    // les floors produits par l'etage 1 de la tentative abandonnee.
    const FloorCounters& c1 = s.counters();
    const bool countersRestored =
        c1.rhoFloored == c0.rhoFloored && c1.uFloored == c0.uFloored &&
        c1.recoveryFailures == c0.recoveryFailures &&
        c1.conservativeResyncs == c0.conservativeResyncs &&
        c1.reconstructionFallbacks == c0.reconstructionFallbacks &&
        c1.coldAtMachinePrecision == c0.coldAtMachinePrecision;
    reportBool("D11e diagnostics 'acceptes' restaures par le rollback", countersRestored);

    // ...mais PAS perdus : la tentative rejetee les a captures.
    const RejectedAttemptTelemetry& tel = s.rejectedTelemetry();
    reportBool("D11f la tentative rejetee est comptee",
               tel.rejectedSteps == rejected0 + 1);
    reportBool("D11g stage-1 floors are retained in rejected telemetry",
               tel.during.totalFloors() > 0);
    reportBool("D11h recovery failure is in rejected telemetry, not accepted counters",
               tel.during.recoveryFailures > 0);
    std::printf("       Accepted: floors=%lld recFail=%lld | Rejected: steps=%lld floors=%lld recFail=%lld\n",
                c1.totalFloors(), c1.recoveryFailures,
                tel.rejectedSteps, tel.during.totalFloors(), tel.during.recoveryFailures);
}

void testForcedRecoveryFailureFallback() {
    Solver1D s(failCfg(RecoveryFailurePolicy::FallbackToPrevious));
    s.initialize([](double x) { return mkPrim(1.0 + 0.2 * std::sin(2.0 * M_PI * x), 1.0, 0.1); });

    const double dt = 0.2 * s.dx();
    s.step(dt);
    const double tBefore = s.time();
    s.injectConservedForTest(Solver1D::NG + 7, inadmissibleState());

    const StepResult r = s.step(dt);

    reportBool("D9a under Fallback, the step continues", r.ok && s.ok());
    reportBool("D9b time advanced", s.time() > tBefore);
    reportBool("D9c failure is counted", s.counters().recoveryFailures > 0);
    reportBool("D9d la non-conservation est comptabilisee",
               s.counters().conservativeResyncs > 0);
    std::printf("       Fallback : recoveryFailures=%lld  conservativeResyncs=%lld  t=%.6g\n",
                s.counters().recoveryFailures, s.counters().conservativeResyncs, s.time());
}

// D10 — une réinitialisation doit rendre l'objet réellement neuf.
void testReinitializationClearsState() {
    Solver1D s(failCfg(RecoveryFailurePolicy::Abort));
    s.initialize([](double x) { return mkPrim(1.0, 1.0, 0.1 * std::sin(x)); });
    s.injectConservedForTest(Solver1D::NG + 3, inadmissibleState());
    s.step(0.2 * s.dx());
    const bool brokenBefore = !s.ok() && s.rejectedTelemetry().during.recoveryFailures > 0;

    // Nouvelle initialisation du MÊME objet.
    s.initialize([](double x) { return mkPrim(1.0 + 0.1 * std::sin(2.0 * M_PI * x), 1.0, 0.05); });

    reportBool("D10a solver was in a failed state before reinitialization", brokenBefore);
    reportBool("D10b initialize() remet ok_ a vrai", s.ok());
    reportBool("D10c initialize() remet les compteurs a zero",
               s.counters().recoveryFailures == 0
               && s.counters().totalFloors() == 0
               && s.counters().conservativeResyncs == 0
               && s.counters().reconstructionFallbacks == 0
               && s.counters().coldAtMachinePrecision == 0);
    reportBool("D10c2 initialize() remet la telemetrie rejetee a zero",
               s.rejectedTelemetry().rejectedSteps == 0
               && s.rejectedTelemetry().during.recoveryFailures == 0);
    reportBool("D10d initialize() resets time to zero", s.time() == 0.0);
    // Et le solveur doit repartir normalement.
    const StepResult r = s.step(0.2 * s.dx());
    reportBool("D10e le solveur reinitialise avance normalement", r.ok && s.ok());
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

// F0b — CONVERGENCE DE L'ORACLE LUI-MÊME.
// Exigence de la revue : ne pas qualifier d'« exact » un résultat dont l'erreur
// propre n'est pas quantifiée. Deux formulations INDÉPENDANTES de la branche de
// détente sont comparées :
//   * la forme CLOSE (dérivée analytiquement, utilisée par le solveur) ;
//   * une quadrature RK4 à N pas, raffinée de 250 à 8000.
// Si la forme close est correcte, l'écart doit décroître en O(N^-4) et tendre
// vers la précision machine — ce qui JUSTIFIE le terme « exact » pour la forme
// close, tout en quantifiant ce que valait l'ancienne version à N=2000.
void testOracleSelfConvergence() {
    const double g = 5.0 / 3.0;
    RiemannState pre{10.0, 13.33, 0.0};
    const double pTarget = 1.4476857;   // ~p* du probleme MM1

    resetRarefactionClampHits();
    std::printf("       convergence de la quadrature RK4 vers la forme CLOSE :\n");
    std::printf("         %-8s %-16s %s\n", "N", "|v_RK4 - v_close|", "ordre");
    const double vClose = rarefactionVelocityAnalytic(pre, pTarget, g, -1);
    double prev = 0.0;
    double bestOrder = 0.0;
    for (int N : {250, 500, 1000, 2000, 4000, 8000}) {
        const double vN = rarefactionVelocityRK4(pre, pTarget, g, -1, N);
        const double e = std::fabs(vN - vClose);
        double order = 0.0;
        if (prev > 0.0 && e > 0.0) order = std::log2(prev / e);
        std::printf("         %-8d %-16.4e %s%.2f\n", N, e, prev > 0.0 ? "" : "  -  ", order);
        if (prev > 1e-15 && e > 1e-15) bestOrder = std::max(bestOrder, order);
        prev = e;
    }
    std::printf("       -> v (forme close) = %.16f\n", vClose);
    reportBool("F0b la quadrature converge vers la forme close (ordre >= 3)",
               bestOrder >= 3.0);
    std::printf("          ordre max observe : %.2f — la forme CLOSE est donc la\n"
                "          limite exacte de l'integration ; c'est elle qu'utilise le\n"
                "          solver, with no residual quadrature error.\n", bestOrder);

    // Impact sur les grandeurs REELLEMENT utilisees ensuite (p*, v*).
    const RiemannState L{10.0, 13.33, 0.0}, R{1.0, 1e-6, 0.0};
    const ExactRiemannSolution sol = solveExactRiemannSRHD(L, R, g);
    std::printf("       -> p* = %.15e   v* = %.15e   (residu %.2e, %d iterations)\n",
                sol.pStar, sol.vStar, sol.residual, sol.iterations);

    reportBool("F0c oracle velocity guard never triggered",
               rarefactionClampHits() == 0,
               rarefactionClampHits() == 0 ? "0 declenchement" : "le clamp a agi (!)");
}

struct ShockTubeResult {
    double l1rho, l1p, l1v;
    long long floors, coldFlags, recoveryFails, reconFallbacks, resyncs;
    bool solverOk;
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
    res.reconFallbacks = s.counters().reconstructionFallbacks;
    res.resyncs = s.counters().conservativeResyncs;
    res.solverOk = s.ok();

    if (verbose) {
        std::printf("         %-22s nx=%-4d t=%.2f CFL=%.2f | L1: rho %.3e  p %.3e  v %.3e\n",
                    label, nx, tEnd, cfl, res.l1rho, res.l1p, res.l1v);
        std::printf("           p*=%.6e v*=%.6e | steps=%d | floors=%lld coldFlag=%lld recFail=%lld\n",
                    sol.pStar, sol.vStar, steps, res.floors, res.coldFlags, res.recoveryFails);
        std::printf("           reconFallbacks=%lld  Uresyncs=%lld  solverOk=%s\n",
                    res.reconFallbacks, res.resyncs, res.solverOk ? "oui" : "NON");
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
    long long totalFloors = 0, totalRecFail = 0, totalFallbacks = 0, totalResyncs = 0;
    bool allSolversOk = true;

    for (const TubeCfg& t : tubes) {
        // Etude de convergence en resolution : l'erreur L1 doit decroitre.
        double prevL1 = 0.0;
        bool decreasing = true;
        for (int nx : {200, 400, 800}) {
            const ShockTubeResult r = runShockTube(t.name, t.L, t.R, t.g, t.tEnd, nx,
                                                   Reconstruction::PLM_MC, 0.4, nx == 800);
            totalFloors += r.floors;
            totalRecFail += r.recoveryFails;
            totalFallbacks += r.reconFallbacks;
            totalResyncs += r.resyncs;
            if (!r.solverOk) allSolversOk = false;
            if (prevL1 > 0.0 && r.l1rho > prevL1) decreasing = false;
            prevL1 = r.l1rho;
            if (!std::isfinite(r.l1rho)) allOk = false;
        }
        if (!decreasing) allOk = false;
        std::printf("           -> L1(rho) decroissante avec la resolution : %s\n",
                    decreasing ? "OUI" : "NON");
    }
    reportBool("F1 tubes a choc : erreurs finies et decroissantes", allOk);
    reportBool("F2 no recovery failure on shock tubes", totalRecFail == 0);
    reportBool("F2b tous les solveurs terminent sans abandon", allSolversOk);
    std::printf("       -> cumul sur tous les tubes/resolutions : floors=%lld  "
                "reconFallbacks=%lld  Uresyncs=%lld\n",
                totalFloors, totalFallbacks, totalResyncs);

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
    reportBool("H1 HDF5 write", wrote);
    if (!wrote) return;

    // Un writer non relu n'est pas un writer teste — et relire seulement
    // certains champs ne teste que ceux-la. On relit TOUT ce qui est ecrit.
    SnapshotContents c;
    const bool read = readSnapshotHDF5(path, c);
    reportBool("H2 relecture HDF5", read);
    if (!read) return;

    reportBool("H3 dimensions coherentes",
               c.nx == cfg.nx && (int)c.rho.size() == cfg.nx
               && (int)c.x.size() == cfg.nx && (int)c.u2.size() == cfg.nx
               && (int)c.u3.size() == cfg.nx);

    // Aller-retour sur TOUS les datasets : RHO, UU, U1, U2, U3 et x.
    double worst = 0.0;
    for (int i = 0; i < cfg.nx; ++i) {
        const PrimitiveState& P = s.primitives()[Solver1D::NG + i];
        auto rel = [](double a, double b) {
            return std::fabs(a - b) / std::max(std::fabs(b), 1e-300);
        };
        worst = std::max(worst, rel(c.rho[i], P.rho));
        worst = std::max(worst, rel(c.uu[i],  P.u));
        worst = std::max(worst, std::fabs(c.u1[i] - P.utilde[0]));
        worst = std::max(worst, std::fabs(c.u2[i] - P.utilde[1]));
        worst = std::max(worst, std::fabs(c.u3[i] - P.utilde[2]));
        worst = std::max(worst, std::fabs(c.x[i] - s.xc(Solver1D::NG + i)));
    }
    report("H4 aller-retour de TOUS les datasets (RHO,UU,U1,U2,U3,x)", worst, 1e-15);
    report("H5 serialized time", std::fabs(c.t - s.time()) / s.time(), 1e-15);
    report("H6 gamma serialise", std::fabs(c.gam - cfg.gamma) / cfg.gamma, 1e-15);

    // Les attributs de provenance doivent survivre : ce sont eux qui
    // empechent de confondre ce fichier avec un vrai snapshot iharm.
    const bool attrsOk = (c.stage == "S3-SRHD-flat")
                      && (c.metric == "minkowski-cartesian-1d")
                      && (c.absentFields.find("B1") != std::string::npos);
    reportBool("H7 attributs de provenance preserves", attrsOk);
    std::printf("       -> bhsim_stage='%s'  metric='%s'\n", c.stage.c_str(), c.metric.c_str());
    std::printf("          absent_fields='%s'\n", c.absentFields.c_str());
    std::printf("       -> NOMS/CONVENTIONS preparant iharm (prims/RHO, UU, U1..U3),\n"
                "          which is NOT full iharm snapshot compatibility :\n"
                "          il manque B1..B3, la metrique/grille FMKS et le spin, et la\n"
                "          geometrie est 1D cartesienne. Ces manques sont declares dans\n"
                "          the file itself, not only in documentation.\n");
}

}  // namespace

// =============================================================================
//  DURCISSEMENT PRE-S4
// =============================================================================

// --- Helpers analytiques, ECRITS ICI et non importes du solveur --------------
// Le principe de la suite est qu'un oracle ne doit rien partager avec ce qu'il
// valide. Ces trois fonctions sont donc reecrites a partir de la definition,
// sans appeler makeFluidVars() ni characteristicSpeeds().
double refEnthalpy(double rho, double p, double g) { return 1.0 + g * p / ((g - 1.0) * rho); }
double refSoundSpeed(double rho, double p, double g) {
    return std::sqrt(g * p / (rho * refEnthalpy(rho, p, g)));
}
double refRiemannG(double c, double g) {
    const double a = std::sqrt(g - 1.0);
    return (2.0 / a) * std::atanh(c / a);
}

// --- I : DOMAINE DE L'ORACLE DE RIEMANN --------------------------------------
void testRiemannVacuumDomain() {
    const double g = 5.0 / 3.0;

    // I1 — Le critere de vide, verifie contre sa LIMITE NEWTONIENNE.
    // Quand c -> 0, g(c) -> 2c/(G-1) et artanh(v) -> v, donc le critere
    // relativiste doit redonner  v_R - v_L >= 2(c_L + c_R)/(G-1)  (Toro §4.6).
    // C'est une verification INDEPENDANTE de la formule : elle vient d'une
    // autre theorie, pas d'une relecture de la meme algebre.
    {
        double worst = 0.0;
        for (double p0 : {1e-10, 1e-8, 1e-6}) {
            const double c = refSoundSpeed(1.0, p0, g);
            const double relat = 2.0 * refRiemannG(c, g);   // membre de droite
            const double newt  = 4.0 * c / (g - 1.0);       // 2(c_L+c_R)/(G-1)
            worst = std::max(worst, std::fabs(relat - newt) / newt);
        }
        report("I1 vacuum criterion -> Toro Newtonian limit", worst, 1e-6);
    }

    // I2 — Le SEUIL est net et le solveur est correct JUSTE EN DESSOUS.
    // On s'approche du seuil par valeurs inferieures : p* doit tendre vers 0
    // continument (les deux detentes se rejoignent de justesse), et la symetrie
    // doit imposer v* = 0.
    {
        const double p0 = 0.01;
        const double c = refSoundSpeed(1.0, p0, g);
        const double vCrit = std::tanh(refRiemannG(c, g));   // seuil symetrique
        bool belowOk = true;
        double lastP = 1e300;
        for (double f : {0.5, 0.8, 0.95, 0.99}) {
            const double v = f * vCrit;
            RiemannState L{1.0, p0, -v}, R{1.0, p0, +v};
            const ExactRiemannSolution s = solveExactRiemannSRHD(L, R, g);
            if (!s.ok || std::fabs(s.vStar) > 1e-12 || !(s.pStar < lastP)) belowOk = false;
            lastP = s.pStar;
        }
        reportBool("I2 sous le seuil : resolu, v*=0, p* decroissant vers 0", belowOk);
        std::printf("       (v_crit symetrique = %.6f ; p* a 0.99*v_crit = %.3e)\n", vCrit, lastP);
    }

    // I3 — ADVERSARIAL : etats fortement divergents -> ECHEC EXPLICITE.
    // C'est le point de la revue : l'oracle doit dire « hors domaine », jamais
    // rendre une fausse solution. Avant durcissement il renvoyait ok=true avec
    // un residu ~1e-15 et un p* SUPERIEUR de deux ordres aux pressions
    // initiales, pour deux etats qui s'ELOIGNENT.
    {
        struct Case { const char* label; RiemannState L, R; };
        const Case cases[] = {
            {"double detente v=+-0.5, p=0.01", {1.0, 0.01, -0.5}, {1.0, 0.01, 0.5}},
            {"double detente v=+-0.9, p=0.01", {1.0, 0.01, -0.9}, {1.0, 0.01, 0.9}},
            {"ultra-relativiste v=+-0.999",    {1.0, 1.0,  -0.999}, {1.0, 1.0, 0.999}},
            {"tiny pressure p=1e-6",         {1.0, 1e-6, -0.9}, {1.0, 1e-6, 0.9}},
            {"asymetrique",                    {2.0, 0.02, -0.8}, {0.5, 0.005, 0.7}},
        };
        bool allRejected = true, allFlagged = true, allDetected = true;
        for (const Case& c : cases) {
            if (!riemannVacuumForms(c.L, c.R, g)) allDetected = false;
            const ExactRiemannSolution s = solveExactRiemannSRHD(c.L, c.R, g);
            if (s.ok) { allRejected = false; std::printf("       (%s : ACCEPTE A TORT p*=%.3e)\n", c.label, s.pStar); }
            if (s.failure != RiemannFailure::VacuumFormed) allFlagged = false;
        }
        reportBool("I3a vacuum formation detected by criterion", allDetected);
        reportBool("I3b oracle refuse de resoudre (ok=false)", allRejected);
        reportBool("I3c cause explicite = VacuumFormed", allFlagged);
    }

    // I4 — Entrees invalides rejetees, sans NaN silencieux.
    {
        bool ok = true;
        const RiemannState good{1.0, 1.0, 0.0};
        const RiemannState bad[] = {
            {-1.0, 1.0, 0.0},                                   // rho < 0
            {1.0, 1.0, 1.5},                                    // |v| > 1
            {std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0},
        };
        for (const RiemannState& b : bad) {
            const ExactRiemannSolution s = solveExactRiemannSRHD(b, good, g);
            if (s.ok || s.failure != RiemannFailure::InvalidInput) ok = false;
        }
        reportBool("I4 entrees invalides -> InvalidInput", ok);
    }
}

// --- J : SYMETRIE DE L'ORACLE (revele le bug du flux de masse non signe) -----
// Une collision frontale symetrique impose v* = 0 EXACTEMENT, par symetrie du
// probleme. Aucune convention de signe n'intervient dans cet enonce : c'est un
// oracle libre de convention, donc un test qu'on ne peut pas « ajuster ».
void testRiemannMirrorSymmetry() {
    const double g = 5.0 / 3.0;

    double worstV = 0.0;
    bool allOk = true;
    for (double v : {0.2, 0.5, 0.9}) {
        for (double p0 : {0.01, 1.0, 100.0}) {
            RiemannState L{1.0, p0, +v}, R{1.0, p0, -v};   // COLLISION
            const ExactRiemannSolution s = solveExactRiemannSRHD(L, R, g);
            if (!s.ok) { allOk = false; continue; }
            worstV = std::max(worstV, std::fabs(s.vStar));
            // La densite post-choc doit elle aussi etre symetrique.
            worstV = std::max(worstV, std::fabs(s.rhoStarL - s.rhoStarR)
                                      / std::max(s.rhoStarL, 1e-30));
        }
    }
    reportBool("J1 toutes les collisions symetriques sont resolues", allOk);
    // DISCRIMINANT : avec le flux de masse NON signe, v* valait ~+0.60 au lieu
    // de 0 (residu 1e-16 : l'oracle etait confiant ET faux).
    report("J2 collision symetrique -> v* = 0 et rho*_L = rho*_R", worstV, 1e-12);

    // J3 — symetrie miroir de la branche de choc elle-meme :
    //   shockBranch((rho,p,-v), p2, -1).vPost  ==  -shockBranch((rho,p,+v), p2, +1).vPost
    double worstS = 0.0;
    for (double v : {0.0, 0.3, 0.5, 0.9}) {
        for (double p2 : {0.5, 2.0, 50.0}) {
            const ShockResult a = shockBranch({1.0, 0.1, -v}, p2, g, -1);
            const ShockResult b = shockBranch({1.0, 0.1, +v}, p2, g, +1);
            if (a.ok != b.ok) { worstS = 1.0; continue; }
            if (!a.ok) continue;
            worstS = std::max(worstS, std::fabs(a.vPost + b.vPost));
            worstS = std::max(worstS, std::fabs(a.shockSpeed + b.shockSpeed));
            worstS = std::max(worstS, std::fabs(a.rhoPost - b.rhoPost));
        }
    }
    report("J3 shock branch: left/right mirror symmetry", worstS, 1e-14);

    // J4 — NON-REGRESSION : les valeurs publiees ne bougent pas.
    // Ces problemes n'exercent que la branche DROITE (leur onde gauche est une
    // detente) : c'est precisement pourquoi ils ne revelaient pas le defaut.
    {
        const ExactRiemannSolution mm1 =
            solveExactRiemannSRHD({10.0, 13.33, 0.0}, {1.0, 1e-6, 0.0}, g);
        const ExactRiemannSolution mm2 =
            solveExactRiemannSRHD({1.0, 1000.0, 0.0}, {1.0, 0.01, 0.0}, g);
        double w = 0.0;
        w = std::max(w, std::fabs(mm1.pStar - 1.4477) / 1.4477);
        w = std::max(w, std::fabs(mm1.vStar - 0.7140) / 0.7140);
        w = std::max(w, std::fabs(mm2.pStar - 18.597) / 18.597);
        w = std::max(w, std::fabs(mm2.vStar - 0.9604) / 0.9604);
        report("J4 MM1/MM2 inchanges vs valeurs publiees", w, 1e-4);
    }
}

// --- K : CONVERGENCE SUR UN MODE ACOUSTIQUE ----------------------------------
// Le test lisse existant transporte un mode CONTACT (p et v uniformes) : il ne
// sollicite ni la pression ni le couplage acoustique. On ajoute donc une onde
// SONORE relativiste de faible amplitude.
//
// SOLUTION LINEARISEE (derivee ici, pas importee). Sur un fond au repos
// (rho0, p0, v=0), en perturbation isentropique  dp = (G p0/rho0) drho :
//   continuite : d_t drho + rho0 d_x dv = 0
//   moment     : rho0 h0 d_t dv + d_x dp = 0
// d'ou  d_tt drho = c_s^2 d_xx drho  avec  c_s^2 = G p0 /(rho0 h0).
// Pour une onde progressive vers +x :  dv = c_s drho/rho0.
//
// INDEPENDANCE : c_s est calcule ici par G p0/(rho0 h0), PAS via
// characteristicSpeeds() -- l'oracle ne doit pas fournir sa propre vitesse.
void testAcousticConvergence() {
    const double g = 4.0 / 3.0;          // gaz relativiste
    const double rho0 = 1.0, p0 = 1.0;
    const double h0 = refEnthalpy(rho0, p0, g);
    const double cs = std::sqrt(g * p0 / (rho0 * h0));
    const double A  = 1e-6;              // amplitude : voir justification infra

    // Le fond doit etre franchement relativiste, sinon on ne teste pas grand-chose.
    reportBool("K0 relativistic sound speed (c_s > 0.4 c)", cs > 0.4);

    auto init = [&](double x) {
        const double sx = std::sin(2.0 * M_PI * x);
        const double rho = rho0 * (1.0 + A * sx);
        const double p   = p0   * (1.0 + g * A * sx);
        const double v   = cs * A * sx;
        return mkPrim(rho, p / (g - 1.0), v);
    };
    // Solution linearisee au temps t : meme profil translate de c_s t.
    auto exactRho = [&](double x, double t) {
        return rho0 * (1.0 + A * std::sin(2.0 * M_PI * (x - cs * t)));
    };

    // Duree : 0.2 periode. L'erreur de LINEARISATION croit comme A^2 * k * c_s * t
    // (raidissement de l'onde simple) ; celle de DISCRETISATION comme A/N^2.
    // Leur rapport vaut ~A N^2, il faut donc A << 1/N^2. Avec N <= 256 et
    // A = 1e-6 : plancher de linearisation ~1.3e-12, tres au-dessous de
    // l'erreur de discretisation attendue (~1e-11 a N=256). Le regime
    // asymptotique est donc bien celui de la DISCRETISATION -- c'est la
    // condition pour que l'ordre mesure ait un sens.
    const double tEnd = 0.2 / cs;

    const int Ns[] = {32, 64, 128, 256};
    double errs[4] = {0, 0, 0, 0};
    double errStatic = 0.0;
    long long clampsBefore = characteristicClampCounters().total();

    for (int k = 0; k < 4; ++k) {
        Solver1DConfig cfg;
        cfg.nx = Ns[k]; cfg.xmin = 0.0; cfg.xmax = 1.0;
        cfg.cfl = 0.4; cfg.gamma = g;
        cfg.recon = Reconstruction::PLM_MC;
        cfg.bc = BoundaryCondition::Periodic;
        cfg.floors.enabled = false;
        cfg.recoveryPolicy = RecoveryFailurePolicy::Abort;

        Solver1D s(cfg);
        s.initialize(init);
        s.advanceTo(tEnd);
        if (!s.ok()) { errs[k] = 1e30; continue; }

        double l1 = 0.0, l1static = 0.0;
        for (int i = Solver1D::NG; i < Solver1D::NG + cfg.nx; ++i) {
            const double x = s.xc(i);
            l1       += std::fabs(s.primitives()[i].rho - exactRho(x, tEnd));
            l1static += std::fabs(s.primitives()[i].rho - exactRho(x, 0.0));
        }
        errs[k] = l1 / cfg.nx;
        if (k == 3) errStatic = l1static / cfg.nx;
    }

    // CONTRE-CONTROLE : l'onde doit REELLEMENT s'etre deplacee a c_s. Si on
    // compare au profil initial (onde immobile), l'erreur doit etre BEAUCOUP
    // plus grande. Sans ce controle, une solution qui ne bouge pas passerait.
    reportBool("K1 l'onde s'est propagee (vs profil immobile)",
               errStatic > 20.0 * errs[3]);
    std::printf("       (L1 vs solution acoustique = %.3e ; vs profil immobile = %.3e)\n",
                errs[3], errStatic);

    double worstOrder = 1e30;
    for (int k = 0; k < 3; ++k) {
        const double order = std::log(errs[k] / errs[k + 1]) / std::log(2.0);
        worstOrder = std::min(worstOrder, order);
        std::printf("       N=%3d -> %3d : L1 = %.4e -> %.4e   ordre = %.3f\n",
                    Ns[k], Ns[k + 1], errs[k], errs[k + 1], order);
    }
    reportBool("K2 ordre >= 1.8 sur TOUS les raffinements (mode acoustique)",
               worstOrder >= 1.8);

    // Le mode acoustique ne doit declencher AUCUN garde-fou.
    reportBool("K3 no characteristic clamp on acoustic wave",
               characteristicClampCounters().total() == clampsBefore);

    // COHERENCE (diagnostic, PAS oracle) : la vitesse du son analytique et
    // lambda_+ du solveur au repos doivent coincider. Rapporte separement pour
    // qu'il soit clair que ce n'est pas ce qui valide la convergence ci-dessus.
    {
        const FluidVars f = makeFluidVars(mkPrim(rho0, p0 / (g - 1.0), 0.0), IdealGasEOS(g));
        const SignalSpeeds sp = characteristicSpeeds(f, 0);
        std::printf("       (diagnostic : c_s analytique = %.10f, lambda+ solveur = %.10f, ecart = %.2e)\n",
                    cs, sp.lambdaPlus, std::fabs(cs - sp.lambdaPlus));
    }
}

// --- L : les garde-fous ne masquent rien dans les tests scientifiques --------
void testClampsNeverFire() {
    const CharacteristicClampCounters& c = characteristicClampCounters();
    std::printf("       cs2Clamped=%lld discNegative=%lld denomDegenerate=%lld lambdaClamped=%lld\n",
                c.cs2Clamped, c.discNegative, c.denomDegenerate, c.lambdaClamped);
    reportBool("L1 no characteristicSpeeds guard activated", c.total() == 0);
    std::printf("       clamp de detente de l'oracle : %lld\n", rarefactionClampHits());
    reportBool("L2 no velocity clamp in rarefaction oracle",
               rarefactionClampHits() == 0);
}

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
    std::printf("\n  D -- conservation globale discrete et contrat sur U\n");
    testGlobalConservation();
    testConservationContract();
    std::printf("\n  D8/D9/D10 -- genuinely forced failure paths\n");
    testForcedRecoveryFailureAbort();
    testAbortRollbackAtStage2();
    testDiagnosticsAreTransactional();
    testForcedRecoveryFailureFallback();
    testReinitializationClearsState();
    std::printf("\n  E -- convergence sur solution LISSE\n");
    testSmoothConvergence();
    std::printf("\n  F -- tubes a choc vs oracle de Riemann EXACT\n");
    testExactSolverJumpConditions();
    testOracleSelfConvergence();
    testShockTubes();
    std::printf("\n  G -- invariance par boost de Lorentz\n");
    testLorentzBoostInvariance();
    std::printf("\n  H -- serialisation HDF5\n");
    testHDF5();
    std::printf("\n  I -- Riemann-oracle domain (vacuum)\n");
    testRiemannVacuumDomain();
    std::printf("\n  J -- symetrie de l'oracle (flux de masse signe)\n");
    testRiemannMirrorSymmetry();
    std::printf("\n  K -- convergence sur un MODE ACOUSTIQUE\n");
    testAcousticConvergence();
    std::printf("\n  L -- numerical guards remained inactive\n");
    testClampsNeverFire();

    std::printf("---------------------------------------------------------------------\n");
    std::printf("  %d tests executes.\n", g_tests);
    if (g_failures == 0)
        std::printf("  S3 VALIDE -- solveur SRHD de reference etabli.\n");
    else
        std::printf("  %d FAILURE(S) -- S3 is NOT validated.\n", g_failures);
    std::printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
