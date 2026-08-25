#include "Fluid.hpp"

#include <algorithm>
#include <cmath>

namespace grmhd {

const char* toString(RecoveryFailure f) {
    switch (f) {
        case RecoveryFailure::None:               return "aucune";
        case RecoveryFailure::NonFiniteInput:     return "entree non finie (NaN/Inf)";
        case RecoveryFailure::NonPositiveDensity: return "D <= 0";
        case RecoveryFailure::EnergyBelowMomentum:return "E <= |S| (etat conserve inadmissible)";
        case RecoveryFailure::BracketFailed:      return "encadrement de la racine impossible";
        case RecoveryFailure::NoConvergence:      return "pas de convergence";
        case RecoveryFailure::UnphysicalResult:   return "resultat hors domaine physique";
        case RecoveryFailure::NegativeInternalEnergy: return "u < 0 (limite de conditionnement)";
    }
    return "inconnue";
}

double lorentzFactor(const PrimitiveState& P) {
    const double u2 = P.utilde[0]*P.utilde[0] + P.utilde[1]*P.utilde[1] + P.utilde[2]*P.utilde[2];
    return std::sqrt(1.0 + u2);
}

// =============================================================================
//  P -> U   (analytique)
// =============================================================================
ConservedState primitiveToConserved(const PrimitiveState& P, const IdealGasEOS& eos) {
    ConservedState U;
    const double W = lorentzFactor(P);
    const double p = eos.pressure(P.rho, P.u);
    const double h = eos.specificEnthalpy(P.rho, P.u);
    const double rhohW2 = P.rho * h * W * W;

    U.D = P.rho * W;
    // S_i = ρ h W² v_i, et v_i = ũ_i / W  (espace plat : indices montés/baissés
    // par δ_ij) => S_i = ρ h W ũ_i.
    for (int i = 0; i < 3; ++i) U.S[i] = P.rho * h * W * P.utilde[i];
    U.tau = rhohW2 - p - U.D;
    return U;
}

// =============================================================================
//  U -> P   (inversion 1D en pression, encadrée + Newton sécurisé)
// =============================================================================
namespace {

// Évalue l'état primitif candidat pour une pression d'essai p, et renvoie le
// résidu f(p) = p_EOS(ρ(p), u(p)) - p dont on cherche la racine.
//
// Chaîne de reconstruction (Valencia) :
//     Z   = E + p        avec E = τ + D          (= ρ h W²)
//     v²  = S²/Z²
//     W   = 1/√(1-v²)
//     ρ   = D/W
//     u   = [τ - D(W-1)]/W² - p(1 - 1/W²)   (forme conditionnée, cf. infra)
struct Trial {
    double rho, u, W, v2, f;
    bool valid;
};

Trial evaluate(double p, double D, double S2, double tau, const IdealGasEOS& eos) {
    Trial t{};
    t.valid = false;

    const double E = tau + D;
    const double Z = E + p;
    if (!(Z > 0.0)) return t;

    const double S = std::sqrt(S2);
    if (Z <= S) return t;                       // impliquerait v >= 1

    t.v2 = S2 / (Z * Z);
    if (!(t.v2 < 1.0)) return t;

    t.W = 1.0 / std::sqrt(1.0 - t.v2);
    if (!std::isfinite(t.W) || t.W < 1.0) return t;

    t.rho = D / t.W;
    if (!(t.rho > 0.0)) return t;

    // ÉNERGIE INTERNE — forme réécrite pour le conditionnement.
    //
    // La forme naïve u = ρ(h - 1 - p/ρ) avec h = Z/(ρW²) soustrait des termes
    // qui contiennent tous la masse au repos : pour ρ=1e6 et u=1e-8, u est
    // perdu dans l'arrondi de ρ. Réécriture algébriquement exacte :
    //
    //   u = ρh - ρ - p = (E+p)/W² - D/W - p ,  puis E = τ + D  donne
    //   u = [τ - D(W-1)]/W² - p(1 - 1/W²)
    //
    // Au repos (W=1) : u = τ EXACTEMENT — la masse au repos n'intervient plus
    // du tout. Il ne subsiste que l'annulation τ - D(W-1), qui est le contenu
    // physique irréductible (énergie au-delà de masse + cinétique).
    t.u = (tau - D * (t.W - 1.0)) / (t.W * t.W) - p * (1.0 - 1.0 / (t.W * t.W));

    t.f = eos.pressure(t.rho, t.u) - p;
    t.valid = std::isfinite(t.f);
    return t;
}

// Dérivée df/dp par différence centrée relative. On reste en différences
// finies plutôt qu'en dérivée analytique : la dérivée sert UNIQUEMENT à
// accélérer Newton, jamais à décider de la convergence (garantie par la
// bisection). Une dérivée légèrement imprécise coûte au pire une itération,
// elle ne peut pas produire un faux résultat.
double dfdp(double p, double D, double S2, double tau, const IdealGasEOS& eos) {
    const double dp = 1e-8 * std::max(std::fabs(p), 1e-10);
    const Trial a = evaluate(p + dp, D, S2, tau, eos);
    const Trial b = evaluate(p - dp, D, S2, tau, eos);
    if (!a.valid || !b.valid) return 0.0;
    return (a.f - b.f) / (2.0 * dp);
}

}  // namespace

bool conservedToPrimitive(const ConservedState& U, const IdealGasEOS& eos,
                          PrimitiveState& P, RecoveryStatus& status) {
    status = RecoveryStatus{};

    // --- 1. Validation de l'état conservé ------------------------------------
    if (!std::isfinite(U.D) || !std::isfinite(U.tau) ||
        !std::isfinite(U.S[0]) || !std::isfinite(U.S[1]) || !std::isfinite(U.S[2])) {
        status.failure = RecoveryFailure::NonFiniteInput;
        return false;
    }
    if (!(U.D > 0.0)) {
        status.failure = RecoveryFailure::NonPositiveDensity;
        return false;
    }

    const double S2 = U.S[0]*U.S[0] + U.S[1]*U.S[1] + U.S[2]*U.S[2];
    const double S  = std::sqrt(S2);
    const double E  = U.tau + U.D;

    // Admissibilité : E > |S| équivaut à v < 1 pour la solution physique.
    // Un état violant cette condition N'EST PAS réparable — on le signale.
    if (!(E > S)) {
        status.failure = RecoveryFailure::EnergyBelowMomentum;
        return false;
    }

    // --- 2. Encadrement de la racine -----------------------------------------
    // Borne basse : p >= 0 (pression positive). Le cas p = 0 correspond au
    // fluide froid, où la racine EST à p = 0 : on l'accepte comme bracket
    // fermé plutôt que de l'exclure par un epsilon arbitraire.
    double plo = 0.0;
    Trial tlo = evaluate(plo, U.D, S2, U.tau, eos);
    if (!tlo.valid) {
        status.failure = RecoveryFailure::BracketFailed;
        return false;
    }

    // Seuil « f(0) est nul aux arrondis près » — à ne PAS tester par `== 0.0` :
    // pour un fluide froid, f(0) vaut typiquement ~1e-17 et peut être
    // légèrement négatif, ce qui ferait rejeter un état parfaitement valide.
    //
    // L'échelle correcte est celle du PLANCHER D'ARRONDI de f, pas celle de
    // l'énergie totale. Avec la forme réécrite, la seule annulation restante
    // est τ - D(W-1), divisée par W² ; f = (Γ-1)u en hérite. Utiliser E ici
    // (comme dans une version antérieure) surestimait le seuil d'un facteur
    // D/u et déclenchait le chemin « froid » sur des fluides tièdes, renvoyant
    // u = 0 — d'où une erreur relative de exactement 1,0.
    // Estimation du plancher d'arrondi des DEUX termes de [τ - D(W-1)] :
    //   * τ  : erreur ~ eps·|τ| ;
    //   * D(W-1) : ATTENTION, l'erreur n'est PAS eps·D(W-1). W est lui-même
    //     entaché d'une erreur relative ~eps, donc (W-1) porte une erreur
    //     ABSOLUE ~eps·W, et le produit une erreur ~eps·D·W. Quand W→1 le
    //     terme D(W-1) tend vers 0 mais son erreur reste ~eps·D : prendre
    //     D(W-1) comme échelle (version antérieure) rendait le seuil trop
    //     serré et faisait rejeter des fluides froids parfaitement valides.
    //   * ET SURTOUT l'amplification par W : 1-v² vaut ~1/W², donc l'arrondi
    //     ABSOLU de v² (~eps) y devient une erreur RELATIVE ~eps·W². W hérite
    //     d'une erreur relative ~½·eps·W², soit une erreur absolue ~½·eps·W³.
    //     Le terme D·W porte alors une erreur ~½·eps·D·W³, qui après division
    //     par W² laisse ~½·eps·D·W. Pour W~18 et D~1e4, ce terme vaut 1,9e-11
    //     et DOMINE de plusieurs ordres les deux précédents : l'ignorer (mes
    //     versions antérieures) sous-estimait le seuil d'un facteur ~80 et
    //     faisait rejeter des états dont u est simplement sous le plancher
    //     numérique de la représentation conservative.
    const double eps = 2.220446049250313e-16;
    const double errScale = (std::fabs(U.tau) + U.D * tlo.W) / (tlo.W * tlo.W)
                          + 0.5 * U.D * tlo.W;
    const double coldTol = 16.0 * eps * (eos.gamma() - 1.0) * errScale;

    // Le raccourci « froid » ne doit s'appliquer QUE si f(0) <= 0, c'est-à-dire
    // quand l'encadrement [0, p_hi] ne peut PAS être formé. Si f(0) > 0, même
    // faiblement, le bracket est valide : on itère normalement et on récupère
    // toute la précision que la double permet. Une version antérieure forçait
    // u = 0 dès que |f(0)| <= coldTol y compris pour f(0) > 0, ce qui jetait
    // de l'information réellement disponible (détecté en comparant à l'oracle
    // long double : écart de rapport 3,6e6 à la borne de conditionnement).
    if (tlo.f <= 0.0) {
        if (tlo.f >= -coldTol) {              // nul aux arrondis près -> froid
            P.rho = tlo.rho;
            P.u = 0.0;
            const double Z = E + plo;
            for (int i = 0; i < 3; ++i) P.utilde[i] = tlo.W * U.S[i] / Z;
            status.converged = true;
            status.residual = std::fabs(tlo.f);
            status.lorentzFactor = tlo.W;
            return true;
        }
        // Franchement négatif : aucune pression positive ne convient, l'état
        // conservé porte moins d'énergie que masse + mouvement.
        status.failure = RecoveryFailure::UnphysicalResult;
        status.residual = std::fabs(tlo.f);
        return false;
    }

    // Borne haute : expansion géométrique jusqu'au changement de signe.
    // f(p) -> (Γ-1)τ - p -> -∞ quand p -> ∞, la borne existe donc toujours
    // pour un état admissible.
    double phi = std::max(1e-10, (eos.gamma() - 1.0) * std::max(U.tau, 1e-30));
    Trial thi = evaluate(phi, U.D, S2, U.tau, eos);
    int expand = 0;
    while ((!thi.valid || thi.f > 0.0) && expand < 200) {
        phi *= 2.0;
        thi = evaluate(phi, U.D, S2, U.tau, eos);
        ++expand;
    }
    if (!thi.valid || thi.f > 0.0) {
        status.failure = RecoveryFailure::BracketFailed;
        status.iterations = expand;
        return false;
    }

    // --- 3. Newton SÉCURISÉ (rtsafe) -----------------------------------------
    // Invariant maintenu : f(plo) > 0 > f(phi). La bisection garantit la
    // convergence, Newton fournit la vitesse. Si Newton sort du bracket ou
    // progresse trop peu, on bisecte — d'où l'impossibilité de diverger.
    double p = 0.5 * (plo + phi);
    double dpOld = std::fabs(phi - plo);
    double dp = dpOld;
    Trial t = evaluate(p, U.D, S2, U.tau, eos);
    if (!t.valid) { p = plo; t = tlo; }

    // Le résidu f est une PRESSION : son échelle naturelle est p lui-même,
    // JAMAIS la densité de masse D ni l'énergie totale. Une version antérieure
    // normalisait par max(|p|, τ, D) : pour ρ=1e6 et u=1e-8, cela tolérait
    // |f| ≤ 1e-8 alors que la racine vaut p ≈ 3,3e-9 — l'erreur admise
    // dépassait la quantité cherchée, et le fluide au repos (cas pourtant
    // exactement soluble) sortait avec 8e-4 d'erreur. Critère RELATIF à p.
    const int maxIter = 200;
    const double tolRel = 1e-14;
    int it = 0;
    for (; it < maxIter; ++it) {
        const double d = dfdp(p, U.D, S2, U.tau, eos);

        const bool newtonOutOfRange =
            ((p - phi) * d - t.f) * ((p - plo) * d - t.f) > 0.0;
        const bool newtonTooSlow = std::fabs(2.0 * t.f) > std::fabs(dpOld * d);

        dpOld = dp;
        if (newtonOutOfRange || newtonTooSlow || d == 0.0) {
            dp = 0.5 * (phi - plo);
            p = plo + dp;
            status.usedBisection = true;
        } else {
            dp = t.f / d;
            p = p - dp;
        }

        Trial tn = evaluate(p, U.D, S2, U.tau, eos);
        if (!tn.valid) {                       // repli sûr : bisection pure
            dp = 0.5 * (phi - plo);
            p = plo + dp;
            tn = evaluate(p, U.D, S2, U.tau, eos);
            status.usedBisection = true;
            if (!tn.valid) {
                status.failure = RecoveryFailure::NoConvergence;
                status.iterations = it + 1;
                return false;
            }
        }
        t = tn;

        // Convergence RELATIVE à la pression elle-même (cf. commentaire above).
        const double pScale = std::max(std::fabs(p), 1e-300);
        if (std::fabs(t.f) <= tolRel * pScale || std::fabs(dp) <= tolRel * pScale) {
            ++it;
            break;
        }

        if (t.f > 0.0) plo = p; else phi = p;
    }

    status.iterations = it;
    status.residual = std::fabs(t.f);

    if (it >= maxIter) {
        status.failure = RecoveryFailure::NoConvergence;
        return false;
    }

    // --- 4. Contrôle du domaine physique du résultat -------------------------
    // AUCUNE correction ici : on vérifie et on signale. Si un plancher doit
    // s'appliquer, ce sera une décision explicite du solveur en S3.
    if (!(t.rho > 0.0) || !(t.v2 < 1.0) || !std::isfinite(t.W) || t.W < 1.0) {
        status.failure = RecoveryFailure::UnphysicalResult;
        status.lorentzFactor = t.W;
        return false;
    }
    if (!(t.u >= 0.0)) {
        // u est reconstruit comme une DIFFÉRENCE entre l'énergie totale et
        // (masse au repos + cinétique). Quand u/ρ << W², cette différence est
        // minuscule devant ses termes et l'arrondi peut la rendre légèrement
        // négative. C'est une limite de CONDITIONNEMENT de la formulation
        // conservative, pas un échec de l'algorithme — on le signale de façon
        // DISTINCTE pour que S3 puisse appliquer sa politique de planchers en
        // connaissance de cause. Aucune correction n'est faite ici.
        status.failure = RecoveryFailure::NegativeInternalEnergy;
        status.lorentzFactor = t.W;
        return false;
    }

    P.rho = t.rho;
    P.u   = t.u;
    const double Z = E + p;
    // v^i = S^i/Z, puis ũ^i = W v^i.
    for (int i = 0; i < 3; ++i) P.utilde[i] = t.W * U.S[i] / Z;

    status.converged = true;
    status.lorentzFactor = t.W;
    return true;
}

}  // namespace grmhd
