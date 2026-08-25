// -----------------------------------------------------------------------------
// S2 — Validation de la conversion primitives <-> conservées (hydro pur).
//
// Leçon de S1 appliquée : un round-trip P->U->P est NÉCESSAIRE mais PAS
// SUFFISANT — il peut s'auto-confirmer si les deux fonctions partagent une
// erreur commune. La suite est donc structurée en trois blocs :
//
//   A. Round-trip massif (déterministe + aléatoire), avec analyse de l'erreur
//      EN FONCTION du régime (W, thermodynamique) plutôt qu'un seuil aveugle.
//   B. Tests INDÉPENDANTS : identités analytiques vérifiées directement sur
//      l'état conservé, formes closes inversibles à la main, et référence
//      `long double` qui ne partage pas l'arithmétique du code testé.
//   C. Limites physiques connues, retrouvées quantitativement.
//
// Aucun plancher correctif : les domaines mal conditionnés sont MESURÉS et
// DOCUMENTÉS, pas masqués.
// -----------------------------------------------------------------------------
#include "Fluid.hpp"
#include "EOS.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace grmhd;

namespace {

int g_failures = 0;
long long g_casesRun = 0;

void report(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  [%s] %-54s  err = %.3e  (tol %.1e)\n",
                ok ? "OK  " : "FAIL", name, err, tol);
    if (!ok) ++g_failures;
}
void reportBool(const char* name, bool ok, const char* detail = nullptr) {
    std::printf("  [%s] %-54s  %s\n", ok ? "OK  " : "FAIL", name, detail ? detail : "");
    if (!ok) ++g_failures;
}

// Générateur déterministe (xorshift) : reproductible d'une exécution à l'autre,
// indispensable pour qu'un échec soit rejouable.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    double uniform() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (s >> 11) * (1.0 / 9007199254740992.0);
    }
    double logUniform(double lo, double hi) {
        return std::exp(std::log(lo) + uniform() * (std::log(hi) - std::log(lo)));
    }
};

double relErr(double a, double b) {
    const double d = std::max(std::fabs(b), 1e-300);
    return std::fabs(a - b) / d;
}

// =============================================================================
//  BLOC A — round-trip massif, avec analyse par régime
// =============================================================================

struct RegimeStats {
    const char* label;
    double worstRho = 0.0, worstU = 0.0, worstVel = 0.0;
    double worstResidual = 0.0;
    int    worstIters = 0;
    long long n = 0, bisections = 0;
    // Distinction ESSENTIELLE : un échec algorithmique (bracket, convergence)
    // est inacceptable ; une limite de conditionnement (u<0 par annulation)
    // est une propriété de la formulation conservative, à documenter.
    long long algoFailures = 0;
    long long conditioningLimited = 0;
    long long byReason[8] = {0,0,0,0,0,0,0,0};   // ventilation par RecoveryFailure
};

void runRoundTrip(RegimeStats& st, const IdealGasEOS& eos,
                  double rhoLo, double rhoHi, double uLo, double uHi,
                  double WLo, double WHi, int nCases, uint64_t seed) {
    Rng rng(seed);
    for (int k = 0; k < nCases; ++k) {
        PrimitiveState P0;
        P0.rho = rng.logUniform(rhoLo, rhoHi);
        P0.u   = rng.logUniform(uLo, uHi);

        // Direction de vitesse arbitraire, uniforme sur la sphère.
        const double W = rng.logUniform(WLo, WHi);
        const double utot = std::sqrt(std::max(W * W - 1.0, 0.0));
        const double cth = 2.0 * rng.uniform() - 1.0;
        const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
        const double ph  = 2.0 * 3.14159265358979323846 * rng.uniform();
        P0.utilde[0] = utot * sth * std::cos(ph);
        P0.utilde[1] = utot * sth * std::sin(ph);
        P0.utilde[2] = utot * cth;

        const ConservedState U = primitiveToConserved(P0, eos);

        PrimitiveState P1;
        RecoveryStatus stt;
        const bool ok = conservedToPrimitive(U, eos, P1, stt);
        ++st.n; ++g_casesRun;
        if (!ok) {
            const int ri = static_cast<int>(stt.failure);
            if (ri >= 0 && ri < 8) ++st.byReason[ri];
            if (st.algoFailures == 0 && st.conditioningLimited == 0) {
                // Capture du PREMIER cas en echec : diagnostic, pas devinette.
                const double W0 = lorentzFactor(P0);
                std::printf("         [1er echec de ce regime] rho=%.6e u=%.6e u/rho=%.2e W=%.6e\n"
                            "            D=%.10e tau=%.10e |S|=%.10e  E-|S|=%.3e  raison=%s\n",
                            P0.rho, P0.u, P0.u / P0.rho, W0, U.D, U.tau,
                            std::sqrt(U.S[0]*U.S[0]+U.S[1]*U.S[1]+U.S[2]*U.S[2]),
                            (U.tau + U.D) - std::sqrt(U.S[0]*U.S[0]+U.S[1]*U.S[1]+U.S[2]*U.S[2]),
                            toString(stt.failure));
            }
            if (stt.failure == RecoveryFailure::NegativeInternalEnergy)
                ++st.conditioningLimited;
            else
                ++st.algoFailures;
            continue;
        }
        if (stt.usedBisection) ++st.bisections;

        st.worstRho = std::max(st.worstRho, relErr(P1.rho, P0.rho));
        st.worstU   = std::max(st.worstU,   relErr(P1.u,   P0.u));
        for (int i = 0; i < 3; ++i) {
            const double scale = std::max(utot, 1e-300);
            st.worstVel = std::max(st.worstVel, std::fabs(P1.utilde[i] - P0.utilde[i]) / scale);
        }
        st.worstResidual = std::max(st.worstResidual, stt.residual);
        st.worstIters = std::max(st.worstIters, stt.iterations);

        // CONTRAINTE DURE : aucun état retourné ne doit violer la physique.
        if (!(P1.rho > 0.0) || !(P1.u >= 0.0) || !std::isfinite(stt.lorentzFactor)
            || stt.lorentzFactor < 1.0) {
            ++st.algoFailures;
        }
    }
}

void testRoundTripByRegime() {
    const IdealGasEOS eos(4.0 / 3.0);
    std::vector<RegimeStats> regimes;

    struct Cfg { const char* label; double rl, rh, ul, uh, wl, wh; };
    const Cfg cfgs[] = {
        {"v faible      (W-1 ~ 1e-8..1e-4)", 1e-6, 1e6, 1e-8, 1e6, 1.0 + 1e-8, 1.0 + 1e-4},
        {"moderement rel(W 1.01..2)       ", 1e-6, 1e6, 1e-8, 1e6, 1.01, 2.0},
        {"relativiste   (W 2..20)         ", 1e-6, 1e6, 1e-8, 1e6, 2.0, 20.0},
        {"ultrarel.     (W 20..1000)      ", 1e-6, 1e6, 1e-8, 1e6, 20.0, 1000.0},
        {"froid         (u/rho ~ 1e-10)   ", 1e-3, 1e3, 1e-13, 1e-7, 1.0 + 1e-6, 50.0},
        {"chaud         (u/rho >> 1)      ", 1e-3, 1e3, 1e3, 1e9, 1.0 + 1e-6, 50.0},
    };
    for (const Cfg& c : cfgs) {
        RegimeStats st; st.label = c.label;
        runRoundTrip(st, eos, c.rl, c.rh, c.ul, c.uh, c.wl, c.wh, 20000, 0xC0FFEEull + regimes.size()*7919);
        regimes.push_back(st);
    }

    std::printf("       round-trip P->U->P par regime (20000 cas chacun) :\n");
    std::printf("         %-34s %-10s %-10s %-10s %-6s %-7s %s\n",
                "regime", "err rho", "err u", "err v", "iters", "ech.algo", "cond.");
    double worstAll = 0.0;
    long long totalAlgo = 0, totalCond = 0;
    for (const RegimeStats& s : regimes) {
        std::printf("         %-34s %-10.2e %-10.2e %-10.2e %-6d %-7lld %lld\n",
                    s.label, s.worstRho, s.worstU, s.worstVel, s.worstIters,
                    s.algoFailures, s.conditioningLimited);
        worstAll = std::max(worstAll, std::max(s.worstRho, s.worstVel));
        totalAlgo += s.algoFailures;
        totalCond += s.conditioningLimited;
    }
    // L'erreur sur `u` est traitée séparément : c'est la quantité intrinsèquement
    // la plus mal conditionnée (voir bloc dédié plus bas).
    // Ventilation des échecs par raison : ne pas se contenter d'un compteur
    // global, il faut savoir SI c'est algorithmique ou du conditionnement.
    long long agg[8] = {0,0,0,0,0,0,0,0};
    for (const RegimeStats& s : regimes)
        for (int i = 0; i < 8; ++i) agg[i] += s.byReason[i];
    bool anyReason = false;
    for (int i = 1; i < 8; ++i) if (agg[i] > 0) anyReason = true;
    if (anyReason) {
        std::printf("       ventilation des echecs par raison :\n");
        for (int i = 1; i < 8; ++i)
            if (agg[i] > 0)
                std::printf("         %-46s %lld\n",
                            toString(static_cast<RecoveryFailure>(i)), agg[i]);
    }

    report("A1 round-trip : rho et vitesse sur tous les regimes", worstAll, 1e-9);
    reportBool("A2a AUCUN echec algorithmique (bracket/convergence)",
               totalAlgo == 0,
               totalAlgo == 0 ? "0 echec algorithmique" : "echecs algorithmiques presents");
    std::printf("       -> %lld cas limites par CONDITIONNEMENT (u<0 par annulation),\n"
                "          signales explicitement et non corriges : c'est a S3 de\n"
                "          decider d'une politique de planchers.\n", totalCond);
}

// A3 — conditionnement de `u` : mise en évidence quantitative.
// Physique du problème : u est reconstruit comme une DIFFÉRENCE entre l'énergie
// totale et (masse au repos + énergie cinétique). Quand u/ρ << W², cette
// différence est petite devant ses termes -> annulation catastrophique
// INHÉRENTE à la formulation, pas un défaut de l'algorithme. On la mesure au
// lieu de la masquer.
void testInternalEnergyConditioning() {
    const IdealGasEOS eos(4.0 / 3.0);
    std::printf("       conditionnement de u : err(u) selon W et u/rho :\n");
    std::printf("         %-12s %-14s %-12s %s\n", "W", "u/rho", "err rel u", "chiffres perdus");

    double worstWellConditioned = 0.0;
    for (double W : {1.0 + 1e-6, 1.1, 10.0, 100.0}) {
        for (double ratio : {1e-8, 1e-4, 1e-1, 1.0, 1e3}) {
            PrimitiveState P0;
            P0.rho = 1.0;
            P0.u = ratio * P0.rho;
            const double utot = std::sqrt(std::max(W * W - 1.0, 0.0));
            P0.utilde[0] = utot; P0.utilde[1] = 0.0; P0.utilde[2] = 0.0;

            const ConservedState U = primitiveToConserved(P0, eos);
            PrimitiveState P1; RecoveryStatus st;
            if (!conservedToPrimitive(U, eos, P1, st)) continue;
            const double e = relErr(P1.u, P0.u);
            const double digits = (e > 0.0) ? -std::log10(std::max(e, 1e-300)) : 16.0;
            std::printf("         %-12.4g %-14.0e %-12.2e ~%.1f chiffres restants\n",
                        W, ratio, e, digits);
            // Régime bien conditionné : gaz tiède/chaud à W modéré.
            if (ratio >= 1e-4 && W <= 10.0) worstWellConditioned = std::max(worstWellConditioned, e);
        }
    }
    report("A3 err(u) dans le regime bien conditionne (u/rho>=1e-4, W<=10)",
           worstWellConditioned, 1e-10);
    std::printf("       -> DOMAINE MAL CONDITIONNE DOCUMENTE (non masque) : fluide tres\n"
                "          froid (u/rho <~ 1e-8) a W eleve. Inherent a la formulation\n"
                "          conservative (u = difference de grands termes), pas un bug.\n");
}

// =============================================================================
//  BLOC B — tests INDÉPENDANTS (ne réutilisent pas primitiveToConserved)
// =============================================================================

// B1 — identités analytiques vérifiées DIRECTEMENT sur l'état conservé.
// On ne compare pas P1 à P0 : on vérifie que les relations structurelles
// reliant D, S, τ, h, W sont satisfaites par le résultat de l'inversion.
void testAnalyticIdentities() {
    const IdealGasEOS eos(5.0 / 3.0);
    Rng rng(0xA11CEull);
    double worstD = 0.0, worstS = 0.0, worstTau = 0.0, worstW = 0.0;

    for (int k = 0; k < 20000; ++k) {
        PrimitiveState P0;
        P0.rho = rng.logUniform(1e-4, 1e4);
        P0.u   = rng.logUniform(1e-4, 1e4);
        const double W = rng.logUniform(1.001, 50.0);
        const double utot = std::sqrt(W * W - 1.0);
        const double cth = 2.0 * rng.uniform() - 1.0;
        const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
        const double ph  = 2.0 * 3.14159265358979323846 * rng.uniform();
        P0.utilde[0] = utot * sth * std::cos(ph);
        P0.utilde[1] = utot * sth * std::sin(ph);
        P0.utilde[2] = utot * cth;

        const ConservedState U = primitiveToConserved(P0, eos);
        PrimitiveState P; RecoveryStatus st;
        if (!conservedToPrimitive(U, eos, P, st)) continue;
        ++g_casesRun;

        // Reconstruction des identités À PARTIR DES PRIMITIVES RETOURNÉES,
        // sans passer par primitiveToConserved.
        const double Wr = std::sqrt(1.0 + P.utilde[0]*P.utilde[0]
                                        + P.utilde[1]*P.utilde[1]
                                        + P.utilde[2]*P.utilde[2]);
        const double p  = (eos.gamma() - 1.0) * P.u;
        const double h  = 1.0 + (P.u + p) / P.rho;      // définition, pas l'EOS helper

        // (i)   D   = ρW
        worstD = std::max(worstD, relErr(P.rho * Wr, U.D));
        // (ii)  S_i = ρ h W ũ_i
        for (int i = 0; i < 3; ++i)
            worstS = std::max(worstS, relErr(P.rho * h * Wr * P.utilde[i], U.S[i]));
        // (iii) τ   = ρhW² - p - D
        worstTau = std::max(worstTau, relErr(P.rho * h * Wr * Wr - p - U.D, U.tau));
        // (iv)  W   = √(1+ũ²)  et cohérence avec le W rapporté par le statut
        worstW = std::max(worstW, relErr(st.lorentzFactor, Wr));
    }
    report("B1a identite  D = rho W", worstD, 1e-12);
    report("B1b identite  S_i = rho h W utilde_i", worstS, 1e-10);
    report("B1c identite  tau = rho h W^2 - p - D", worstTau, 1e-9);
    report("B1d coherence du facteur de Lorentz rapporte", worstW, 1e-14);
}

// B2 — formes CLOSES inversibles à la main : oracle totalement indépendant.
void testClosedFormCases() {
    const IdealGasEOS eos(4.0 / 3.0);

    // (a) Fluide au repos : W=1 => D=ρ et τ=u EXACTEMENT (résultat analytique).
    //     On construit U à la main, sans primitiveToConserved.
    double worstRest = 0.0;
    for (double rho : {1e-6, 1e-2, 1.0, 1e3, 1e6}) {
        for (double u : {1e-8, 1e-2, 1.0, 1e4}) {
            ConservedState U;
            U.D = rho; U.tau = u;
            U.S[0] = U.S[1] = U.S[2] = 0.0;
            PrimitiveState P; RecoveryStatus st;
            if (!conservedToPrimitive(U, eos, P, st)) { worstRest = 1.0; continue; }
            worstRest = std::max(worstRest, relErr(P.rho, rho));
            worstRest = std::max(worstRest, relErr(P.u, u));
            worstRest = std::max(worstRest, std::fabs(P.utilde[0]) + std::fabs(P.utilde[1]) + std::fabs(P.utilde[2]));
        }
    }
    report("B2a fluide au repos : D=rho, tau=u (forme close)", worstRest, 1e-12);

    // (b) Fluide FROID (u=0, p=0) : h=1, donc
    //        D = ρW ,  S = ρW ũ ,  τ = ρW(W-1) = D(W-1)
    //     => ũ = |S|/D et W = √(1+ũ²) s'inversent EXPLICITEMENT.
    //     Oracle purement algébrique, aucune itération.
    double worstCold = 0.0;
    for (double rho : {1e-3, 1.0, 1e3}) {
        for (double Wt : {1.0 + 1e-9, 1.5, 10.0, 200.0}) {
            const double ut = std::sqrt(std::max(Wt * Wt - 1.0, 0.0));
            ConservedState U;
            U.D = rho * Wt;
            U.S[0] = rho * Wt * ut; U.S[1] = 0.0; U.S[2] = 0.0;
            U.tau = U.D * (Wt - 1.0);
            PrimitiveState P; RecoveryStatus st;
            if (!conservedToPrimitive(U, eos, P, st)) { worstCold = 1.0; continue; }
            worstCold = std::max(worstCold, relErr(P.rho, rho));
            worstCold = std::max(worstCold, relErr(st.lorentzFactor, Wt));
            // u doit être nul (ou négligeable devant ρ).
            worstCold = std::max(worstCold, P.u / rho);
        }
    }
    report("B2b fluide froid : inversion algebrique explicite", worstCold, 1e-9);
}

// B3 — référence LONG DOUBLE : l'oracle ne partage pas l'arithmétique du code.
// On réimplémente l'inversion en `long double` avec une bisection pure (donc
// un algorithme DIFFÉRENT : pas de Newton, pas de dérivée numérique) et on
// compare. Un bug algébrique commun serait révélé ; une simple différence
// d'arrondi ne l'est pas.
bool invertLongDouble(const ConservedState& U, long double gamma,
                      long double& rho, long double& u, long double& W) {
    const long double D = U.D;
    const long double S2 = (long double)U.S[0]*U.S[0]
                         + (long double)U.S[1]*U.S[1]
                         + (long double)U.S[2]*U.S[2];
    const long double S = sqrtl(S2);
    const long double E = (long double)U.tau + D;
    if (!(E > S) || !(D > 0.0L)) return false;

    auto f = [&](long double p, long double& rr, long double& uu, long double& WW) -> long double {
        const long double Z = E + p;
        if (Z <= S) return -1e300L;
        const long double v2 = S2 / (Z * Z);
        if (!(v2 < 1.0L)) return -1e300L;
        WW = 1.0L / sqrtl(1.0L - v2);
        rr = D / WW;
        const long double h = Z / (rr * WW * WW);
        const long double eps = h - 1.0L - p / rr;
        uu = rr * eps;
        return (gamma - 1.0L) * uu - p;
    };

    long double lo = 0.0L, hi = fmaxl(1e-30L, (gamma - 1.0L) * fmaxl((long double)U.tau, 1e-30L));
    long double rr, uu, WW;
    int guard = 0;
    while (f(hi, rr, uu, WW) > 0.0L && guard++ < 300) hi *= 2.0L;
    if (f(lo, rr, uu, WW) < 0.0L) return false;

    // Bisection pure : 200 itérations => bien au-delà de la précision du type.
    for (int i = 0; i < 200; ++i) {
        const long double mid = 0.5L * (lo + hi);
        if (f(mid, rr, uu, WW) > 0.0L) lo = mid; else hi = mid;
    }
    f(0.5L * (lo + hi), rho, u, W);
    return true;
}

void testLongDoubleReference() {
    const IdealGasEOS eos(4.0 / 3.0);
    Rng rng(0xBEEF1234ull);
    double worstRho = 0.0, worstU = 0.0, worstW = 0.0;
    double worstUillCond = 0.0;
    int compared = 0, illConditioned = 0;

    for (int k = 0; k < 5000; ++k) {
        PrimitiveState P0;
        P0.rho = rng.logUniform(1e-3, 1e3);
        P0.u   = rng.logUniform(1e-3, 1e3);
        const double W = rng.logUniform(1.001, 100.0);
        const double utot = std::sqrt(W * W - 1.0);
        const double cth = 2.0 * rng.uniform() - 1.0;
        const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
        const double ph  = 2.0 * 3.14159265358979323846 * rng.uniform();
        P0.utilde[0] = utot * sth * std::cos(ph);
        P0.utilde[1] = utot * sth * std::sin(ph);
        P0.utilde[2] = utot * cth;

        const ConservedState U = primitiveToConserved(P0, eos);

        PrimitiveState P; RecoveryStatus st;
        if (!conservedToPrimitive(U, eos, P, st)) continue;

        long double lrho, lu, lW;
        if (!invertLongDouble(U, (long double)eos.gamma(), lrho, lu, lW)) continue;
        ++compared; ++g_casesRun;

        worstRho = std::max(worstRho, relErr(P.rho, (double)lrho));
        worstW   = std::max(worstW,   relErr(st.lorentzFactor, (double)lW));

        // Au lieu de découper arbitrairement en « bon » et « mauvais » régime,
        // on teste que l'erreur RESPECTE LA BORNE DE CONDITIONNEMENT PRÉDITE
        // — test bien plus fort, valable sur TOUS les régimes.
        //
        // Analyse de propagation. u est reconstruit par
        //     u = [τ - D(W-1)]/W² - p(1 - 1/W²)
        // L'erreur dominante vient de W : comme 1-v² ~ 1/W², l'arrondi ABSOLU
        // de v² (~eps) y devient une erreur relative ~eps·W², d'où
        //     δW ~ ½·eps·W³.
        // En dérivant,  ∂u/∂W ≈ -D/W² - 2u/W , donc
        //     δu ≈ (D/W² + 2u/W)·δW = ½·eps·(D·W + 2u·W²)
        // soit, en relatif (avec D·W/u = ρW²/u ≡ κ) :
        //     δu/u ≈ ½·eps·(κ + 2W²)
        //
        // ⚠️ Le terme 2W² est ESSENTIEL et manquait dans une première version
        // du modèle : il subsiste même pour κ → 0 (fluide très chaud), où la
        // borne ½·eps·κ seule descendait SOUS l'epsilon machine — une borne
        // qu'aucun calcul ne peut respecter. Symptôme : des rapports
        // observé/prédit de 3,6e6 sur des cas pourtant parfaitement sains.
        const double eU = relErr(P.u, (double)lu);
        const double W0 = std::sqrt(1.0 + P0.utilde[0]*P0.utilde[0]
                                        + P0.utilde[1]*P0.utilde[1]
                                        + P0.utilde[2]*P0.utilde[2]);
        const double kappa = P0.rho * W0 * W0 / std::max(P0.u, 1e-300);
        const double predicted = 0.5 * 2.220446049250313e-16
                               * (kappa + 2.0 * W0 * W0);
        const double ratio = eU / std::max(predicted, 1e-300);
        worstU = std::max(worstU, ratio);
        if (kappa > 1e10) { worstUillCond = std::max(worstUillCond, eU); ++illConditioned; }
    }
    std::printf("       %d cas compares a la reference long double (bisection pure,\n"
                "       algebre naive conservee pour l'independance)\n", compared);
    report("B3a rho vs reference long double", worstRho, 1e-11);
    report("B3c W   vs reference long double", worstW,   1e-11);
    // Le critere porte sur le RAPPORT observe/predit : l'erreur doit rester
    // dans un petit multiple de la borne de conditionnement, sur TOUS les cas.
    // Le facteur 50 couvre la tolérance de convergence du solveur (1e-14 ~ 45 eps)
    // et l'accumulation sur les quelques opérations de la chaîne.
    report("B3b u : ecart / borne de conditionnement predite", worstU, 50.0);
    std::printf("       -> rapport max observe/predit = %.2f  (borne 1/2*eps*(kappa+2W^2))\n"
                "          %d cas a kappa > 1e10 (u sous le plancher numerique de la\n"
                "          representation conservative), ecart max sur u = %.2e\n",
                worstU, illConditioned, worstUillCond);
}

// =============================================================================
//  BLOC C — limites physiques connues
// =============================================================================

// C1 — limite newtonienne : pour v << 1 et u << ρ,
//        τ -> u + ½ ρ v²   (énergie interne + cinétique)
// On vérifie que la relation est retrouvée QUANTITATIVEMENT, à l'ordre attendu.
void testNewtonianLimit() {
    const IdealGasEOS eos(5.0 / 3.0);
    std::printf("       limite newtonienne : tau vs (u + 1/2 rho v^2) :\n");
    double worstScaled = 0.0;
    for (double v : {1e-4, 1e-3, 1e-2}) {
        PrimitiveState P;
        P.rho = 1.0; P.u = 1e-6;
        const double W = 1.0 / std::sqrt(1.0 - v * v);
        P.utilde[0] = W * v; P.utilde[1] = 0.0; P.utilde[2] = 0.0;
        const ConservedState U = primitiveToConserved(P, eos);
        const double newtonian = P.u + 0.5 * P.rho * v * v;
        const double rel = std::fabs(U.tau - newtonian) / newtonian;
        // L'écart doit être O(v²) relatif : on normalise pour vérifier l'ORDRE.
        const double scaled = rel / (v * v);
        std::printf("         v = %-8.0e  tau = %-12.6e  newtonien = %-12.6e  ecart/v^2 = %.3f\n",
                    v, U.tau, newtonian, scaled);
        worstScaled = std::max(worstScaled, scaled);
    }
    // Le rapport doit rester BORNÉ (O(1)) quand v -> 0 : preuve que l'écart
    // est bien d'ordre v² et non d'ordre inférieur.
    report("C1 limite newtonienne : ecart d'ordre v^2 (rapport borne)", worstScaled, 5.0);
}

// C2 — états inadmissibles : doivent être REJETÉS avec un diagnostic, jamais
// « réparés » silencieusement.
void testRejectionOfInvalidStates() {
    const IdealGasEOS eos(4.0 / 3.0);
    PrimitiveState P; RecoveryStatus st;
    bool allRejected = true;
    struct Case { const char* label; ConservedState U; RecoveryFailure expect; };

    ConservedState u1; u1.D = -1.0; u1.tau = 1.0;
    ConservedState u2; u2.D = 1.0;  u2.tau = 1.0; u2.S[0] = 100.0;   // |S| >> E
    ConservedState u3; u3.D = std::nan(""); u3.tau = 1.0;
    ConservedState u4; u4.D = 1.0; u4.tau = -0.9999;                  // E < D, pas d'energie

    const Case cases[] = {
        {"densite negative D<0",            u1, RecoveryFailure::NonPositiveDensity},
        {"moment > energie (|S|>E)",        u2, RecoveryFailure::EnergyBelowMomentum},
        {"entree NaN",                      u3, RecoveryFailure::NonFiniteInput},
        {"energie insuffisante",            u4, RecoveryFailure::UnphysicalResult},
    };
    std::printf("       rejet des etats conserves inadmissibles :\n");
    for (const Case& c : cases) {
        const bool ok = conservedToPrimitive(c.U, eos, P, st);
        const bool rejected = !ok;
        std::printf("         %-32s -> %s (%s)\n", c.label,
                    rejected ? "REJETE" : "ACCEPTE (!!)", toString(st.failure));
        if (!rejected) allRejected = false;
    }
    reportBool("C2 tous les etats inadmissibles sont rejetes, non repares", allRejected);
}

// C3 — stress massif à la frontière d'admissibilité (E légèrement > |S|),
// là où v -> 1 et où l'inversion est la plus sollicitée.
void testNearAdmissibilityBoundary() {
    const IdealGasEOS eos(4.0 / 3.0);
    long long n = 0, ok = 0, unphysical = 0;
    int maxIters = 0;
    double worstResidual = 0.0;

    for (double eps : {1e-3, 1e-6, 1e-9, 1e-12}) {
        for (double rho : {1e-3, 1.0, 1e3}) {
            for (double Wt : {10.0, 100.0, 1000.0}) {
                // Fluide quasi froid à W élevé : E n'excède |S| que de peu.
                const double ut = std::sqrt(Wt * Wt - 1.0);
                PrimitiveState P0;
                P0.rho = rho; P0.u = eps * rho;
                P0.utilde[0] = ut; P0.utilde[1] = 0.0; P0.utilde[2] = 0.0;
                const ConservedState U = primitiveToConserved(P0, eos);

                PrimitiveState P; RecoveryStatus st;
                ++n; ++g_casesRun;
                if (conservedToPrimitive(U, eos, P, st)) {
                    ++ok;
                    maxIters = std::max(maxIters, st.iterations);
                    worstResidual = std::max(worstResidual, st.residual);
                    if (!(P.rho > 0.0) || !(P.u >= 0.0) || st.lorentzFactor < 1.0)
                        ++unphysical;
                }
            }
        }
    }
    std::printf("       frontiere d'admissibilite : %lld cas, %lld converges, "
                "%d iterations max, residu max %.2e\n", n, ok, maxIters, worstResidual);
    reportBool("C3a aucun etat non physique retourne pres de la frontiere",
               unphysical == 0);
    reportBool("C3b convergence sur toute la frontiere testee", ok == n);
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf("  S2 : primitives <-> conservees, HYDRODYNAMIQUE RELATIVISTE PURE\n");
    std::printf("       (aucun champ magnetique a ce stade)\n");
    std::printf("=====================================================================\n");

    std::printf("\n  BLOC A -- round-trip massif et conditionnement\n");
    testRoundTripByRegime();
    testInternalEnergyConditioning();

    std::printf("\n  BLOC B -- oracles INDEPENDANTS\n");
    testAnalyticIdentities();
    testClosedFormCases();
    testLongDoubleReference();

    std::printf("\n  BLOC C -- limites physiques et robustesse\n");
    testNewtonianLimit();
    testRejectionOfInvalidStates();
    testNearAdmissibilityBoundary();

    std::printf("---------------------------------------------------------------------\n");
    std::printf("  %lld cas testes au total.\n", g_casesRun);
    if (g_failures == 0)
        std::printf("  S2 VALIDE -- inversion hydro robuste, S3 peut demarrer.\n");
    else
        std::printf("  %d ECHEC(S) -- S2 NON valide, ne pas demarrer S3.\n", g_failures);
    std::printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
