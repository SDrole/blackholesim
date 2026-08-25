#include "Solver1D.hpp"
#include "Riemann.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace grmhd {

namespace {

// --- Limiteurs de pente ------------------------------------------------------
double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::fabs(a) < std::fabs(b)) ? a : b;
}
// MC (monotonized central) : moins diffusif que minmod, toujours TVD.
double mcLimiter(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    const double c = 0.5 * (a + b);
    const double s = (a > 0.0) ? 1.0 : -1.0;
    return s * std::min(std::min(2.0 * std::fabs(a), 2.0 * std::fabs(b)), std::fabs(c));
}

}  // namespace

Solver1D::Solver1D(const Solver1DConfig& cfg)
    : cfg_(cfg), eos_(cfg.gamma) {
    dx_ = (cfg_.xmax - cfg_.xmin) / cfg_.nx;
    prim_.resize(total());
    cons_.resize(total());
}

void Solver1D::applyBoundaryPrimitives() {
    const int n = cfg_.nx;
    if (cfg_.bc == BoundaryCondition::Periodic) {
        for (int g = 0; g < NG; ++g) {
            prim_[g]              = prim_[n + g];
            prim_[NG + n + g]     = prim_[NG + g];
        }
    } else {  // Outflow : gradient nul
        for (int g = 0; g < NG; ++g) {
            prim_[g]          = prim_[NG];
            prim_[NG + n + g] = prim_[NG + n - 1];
        }
    }
}

void Solver1D::applyBoundaryConserved() {
    const int n = cfg_.nx;
    if (cfg_.bc == BoundaryCondition::Periodic) {
        for (int g = 0; g < NG; ++g) {
            cons_[g]          = cons_[n + g];
            cons_[NG + n + g] = cons_[NG + g];
        }
    } else {
        for (int g = 0; g < NG; ++g) {
            cons_[g]          = cons_[NG];
            cons_[NG + n + g] = cons_[NG + n - 1];
        }
    }
}

double Solver1D::computeTimeStep() const {
    double maxSpeed = 0.0;
    for (int i = NG; i < NG + cfg_.nx; ++i) {
        const FluidVars f = makeFluidVars(prim_[i], eos_);
        const SignalSpeeds s = characteristicSpeeds(f, 0);
        maxSpeed = std::max(maxSpeed, std::max(std::fabs(s.lambdaMinus),
                                                std::fabs(s.lambdaPlus)));
    }
    // Plancher : évite dt infini pour un état strictement au repos et froid.
    maxSpeed = std::max(maxSpeed, 1e-10);
    return cfg_.cfl * dx_ / maxSpeed;
}

// POINT UNIQUE d'application de la politique de recovery.
// Toute conversion U→P du solveur passe par ici : il ne doit pas exister deux
// implémentations du contrat susceptibles de diverger (c'est précisément ce
// qui s'était produit — computeRHS repartait d'une primitive périmée même
// sous la politique Abort, contournant le contrat annoncé).
bool Solver1D::recoverWithPolicy(int cell, const ConservedState& U, const PrimitiveState& fallback,
                                 PrimitiveState& out, RecoveryStatus& st,
                                 bool& usedFallback) {
    usedFallback = false;

    // Injection de faute (tests uniquement). Placée ICI et nulle part ailleurs :
    // une faute injectée doit emprunter exactement le même chemin de politique
    // qu'un échec réel, sinon le test validerait un chemin qui n'existe pas.
    const bool forced = (faultStage_ != 0 && faultStage_ == curStage_ && faultCell_ == cell);

    const bool ok = forced ? false : conservedToPrimitive(U, eos_, out, st);
    if (forced) st.failure = RecoveryFailure::NoConvergence;
    if (st.coldAtMachinePrecision) ++counters_.coldAtMachinePrecision;
    if (ok) return true;

    ++counters_.recoveryFailures;
    if (cfg_.recoveryPolicy == RecoveryFailurePolicy::Abort) return false;

    // FallbackToPrevious UNIQUEMENT : réparation non conservative, assumée.
    out = fallback;
    usedFallback = true;
    return true;
}

bool Solver1D::computeRHS(const std::vector<ConservedState>& cons,
                          std::vector<std::array<double, 5>>& rhs, StepResult& res) {
    const int nTot = total();

    // 1) Primitives locales (les conservées font foi).
    // On convertit TOUT avant de calculer le moindre flux : sous Abort, un
    // échec interrompt ici, donc aucun flux ne peut être bâti sur un état
    // non récupéré.
    static thread_local std::vector<PrimitiveState> P;
    static thread_local std::vector<FluidVars> FV;
    P.resize(nTot); FV.resize(nTot);
    for (int i = 0; i < nTot; ++i) {
        RecoveryStatus st;
        PrimitiveState p;
        bool usedFallback = false;
        if (!recoverWithPolicy(i, cons[i], prim_[i], p, st, usedFallback)) {
            res.ok = false;
            res.failure = st.failure;
            res.failedCell = i;
            return false;
        }
        P[i] = p;
        FV[i] = makeFluidVars(P[i], eos_);
    }

    // 2) Reconstruction aux faces + flux de Riemann.
    // UNE SEULE évaluation par face, partagée par les deux cellules adjacentes :
    // c'est ce qui rend le schéma conservatif au niveau de l'arrondi.
    static thread_local std::vector<std::array<double, 5>> faceFlux;
    faceFlux.resize(nTot + 1);

    auto reconstructPair = [&](int i, PrimitiveState& left, PrimitiveState& right) {
        // Face entre i et i+1 : état gauche = extrapolation depuis i,
        // état droit = extrapolation depuis i+1.
        if (cfg_.recon == Reconstruction::PiecewiseConstant) {
            left = P[i]; right = P[i + 1];
            return;
        }
        auto lim = (cfg_.recon == Reconstruction::PLM_MinMod) ? minmod : mcLimiter;
        auto slope = [&](int j, double PrimitiveState::* /*unused*/) { (void)j; return 0.0; };
        (void)slope;

        // Pente limitée composante par composante, sur les PRIMITIVES
        // (reconstruire les primitives évite de produire des états conservés
        // inadmissibles, contrairement à une reconstruction des conservées).
        auto doSlope = [&](double a, double b) { return lim(a, b); };

        auto recon1 = [&](int j, double vm, double v0, double vp, double& lo, double& hi) {
            const double s = doSlope(v0 - vm, vp - v0);
            lo = v0 + 0.5 * s;    // extrapolation vers la face droite de j
            hi = v0 - 0.5 * s;    // extrapolation vers la face gauche de j
            (void)j;
        };

        double lo, hi;
        recon1(i, P[i-1].rho, P[i].rho, P[i+1].rho, lo, hi);   left.rho  = lo;
        recon1(i, P[i-1].u,   P[i].u,   P[i+1].u,   lo, hi);   left.u    = lo;
        for (int k = 0; k < 3; ++k) {
            recon1(i, P[i-1].utilde[k], P[i].utilde[k], P[i+1].utilde[k], lo, hi);
            left.utilde[k] = lo;
        }
        recon1(i+1, P[i].rho, P[i+1].rho, P[i+2].rho, lo, hi); right.rho = hi;
        recon1(i+1, P[i].u,   P[i+1].u,   P[i+2].u,   lo, hi); right.u   = hi;
        for (int k = 0; k < 3; ++k) {
            recon1(i+1, P[i].utilde[k], P[i+1].utilde[k], P[i+2].utilde[k], lo, hi);
            right.utilde[k] = hi;
        }

        // Garde-fou de positivité de la RECONSTRUCTION (distinct des floors
        // physiques) : si l'extrapolation produit ρ ou u négatif, on retombe
        // localement au premier ordre. C'est la pratique standard et cela
        // reste une décision de la brique « reconstruction », pas un floor
        // dissimulé sur l'état physique.
        if (left.rho  <= 0.0 || left.u  < 0.0) { left  = P[i];     ++counters_.reconstructionFallbacks; }
        if (right.rho <= 0.0 || right.u < 0.0) { right = P[i + 1]; ++counters_.reconstructionFallbacks; }
    };

    const int loFace = (cfg_.recon == Reconstruction::PiecewiseConstant) ? NG - 1 : NG - 1;
    const int hiFace = NG + cfg_.nx;
    for (int i = loFace; i < hiFace; ++i) {
        PrimitiveState pl = P[i], pr = P[i + 1];
        if (cfg_.recon != Reconstruction::PiecewiseConstant && i >= 1 && i + 2 < nTot)
            reconstructPair(i, pl, pr);

        RiemannInput in;
        in.fL = makeFluidVars(pl, eos_);
        in.fR = makeFluidVars(pr, eos_);
        in.uL = primitiveToConserved(pl, eos_);
        in.uR = primitiveToConserved(pr, eos_);
        hlleFlux(in, 0, faceFlux[i + 1].data());   // flux de la face entre i et i+1
    }

    // 3) Divergence : chaque cellule utilise EXACTEMENT le même tableau de
    // flux que sa voisine -> télescopage exact des sommes globales.
    rhs.assign(nTot, {0.0, 0.0, 0.0, 0.0, 0.0});
    const double invdx = 1.0 / dx_;
    for (int i = NG; i < NG + cfg_.nx; ++i)
        for (int k = 0; k < 5; ++k)
            rhs[i][k] = -(faceFlux[i + 1][k] - faceFlux[i][k]) * invdx;
    return true;
}

bool Solver1D::recoverPrimitives(std::vector<ConservedState>& cons, StepResult& res) {
    for (int i = NG; i < NG + cfg_.nx; ++i) {
        RecoveryStatus st;
        PrimitiveState p;
        // `modified` trace si la primitive a été RÉELLEMENT changée par une
        // politique explicite. C'est la seule condition qui autorise à
        // réécrire U (contrat de conservation).
        bool modified = false;
        bool usedFallback = false;

        if (!recoverWithPolicy(i, cons[i], prim_[i], p, st, usedFallback)) {
            res.ok = false;
            res.failure = st.failure;
            res.failedCell = i;
            return false;               // Abort : on n'invente aucun état
        }
        if (usedFallback) modified = true;   // réparation NON CONSERVATIVE

        // --- FLOORS : couche explicite, appliquée APRÈS le recovery ---------
        if (cfg_.floors.enabled) {
            counters_.minRhoBeforeFloor = std::min(counters_.minRhoBeforeFloor, p.rho);
            counters_.minUBeforeFloor   = std::min(counters_.minUBeforeFloor, p.u);
            if (p.rho < cfg_.floors.rhoMin) {
                p.rho = cfg_.floors.rhoMin; ++counters_.rhoFloored; modified = true;
            }
            if (p.u < cfg_.floors.uMin) {
                p.u = cfg_.floors.uMin; ++counters_.uFloored; modified = true;
            }
        }

        prim_[i] = p;

        // --- CONTRAT DE CONSERVATION ----------------------------------------
        // U est la variable ÉVOLUÉE par le schéma volumes finis : elle fait
        // foi. Tant que le recovery réussit et qu'aucune politique n'a modifié
        // P, U doit rester EXACTEMENT tel que produit par la mise à jour
        // conservative — pas un aller-retour P->U->P.
        //
        // Une version antérieure resynchronisait systématiquement
        // (`cons[i] = primitiveToConserved(p, eos_)`), ce qui injectait à
        // chaque sous-étage l'erreur d'arrondi de l'aller-retour dans la
        // variable conservée et affaiblissait la propriété de conservation
        // que tout le schéma est censé garantir.
        //
        // On ne resynchronise donc QUE si P a réellement changé — auquel cas
        // la conservation est localement rompue par construction, et c'est
        // comptabilisé.
        if (modified) {
            cons[i] = primitiveToConserved(p, eos_);
            ++counters_.conservativeResyncs;
        }
    }
    return true;
}

StepResult Solver1D::step(double dt) {
    StepResult res;
    const int nTot = total();
    std::vector<std::array<double, 5>> rhs;

    // INSTANTANÉ TRANSACTIONNEL : U et P au début du pas. En cas d'abandon,
    // on restaure ce couple exact — un état à moitié avancé avec un `t_`
    // inchangé ne correspondrait à aucun temps physique, et interdirait de
    // réessayer avec un dt plus petit.
    const std::vector<ConservedState> u0 = cons_;
    const std::vector<PrimitiveState> p0 = prim_;
    auto rollback = [&]() { cons_ = u0; prim_ = p0; };

    auto addRHS = [&](std::vector<ConservedState>& dst,
                      const std::vector<ConservedState>& src,
                      const std::vector<std::array<double, 5>>& r, double a) {
        for (int i = NG; i < NG + cfg_.nx; ++i) {
            dst[i].D    = src[i].D    + a * r[i][0];
            dst[i].S[0] = src[i].S[0] + a * r[i][1];
            dst[i].S[1] = src[i].S[1] + a * r[i][2];
            dst[i].S[2] = src[i].S[2] + a * r[i][3];
            dst[i].tau  = src[i].tau  + a * r[i][4];
        }
    };

    // --- Étage 1 -------------------------------------------------------------
    curStage_ = 1;
    applyBoundaryConserved();
    if (!computeRHS(cons_, rhs, res)) { res.failedStage = 1; ok_ = false; rollback(); return res; }
    std::vector<ConservedState> u1 = cons_;
    addRHS(u1, u0, rhs, dt);
    cons_ = u1;
    if (!recoverPrimitives(cons_, res)) { res.failedStage = 1; ok_ = false; rollback(); return res; }
    applyBoundaryPrimitives();
    applyBoundaryConserved();

    // --- Étage 2 (moyenne SSP) -----------------------------------------------
    curStage_ = 2;
    if (!computeRHS(cons_, rhs, res)) { res.failedStage = 2; ok_ = false; rollback(); return res; }
    std::vector<ConservedState> u2 = cons_;
    addRHS(u2, cons_, rhs, dt);
    for (int i = NG; i < NG + cfg_.nx; ++i) {
        cons_[i].D    = 0.5 * (u0[i].D    + u2[i].D);
        cons_[i].S[0] = 0.5 * (u0[i].S[0] + u2[i].S[0]);
        cons_[i].S[1] = 0.5 * (u0[i].S[1] + u2[i].S[1]);
        cons_[i].S[2] = 0.5 * (u0[i].S[2] + u2[i].S[2]);
        cons_[i].tau  = 0.5 * (u0[i].tau  + u2[i].tau);
    }
    if (!recoverPrimitives(cons_, res)) { res.failedStage = 2; ok_ = false; rollback(); return res; }
    applyBoundaryPrimitives();
    applyBoundaryConserved();

    // Le temps n'avance QUE si le pas est allé au bout.
    t_ += dt;
    (void)nTot;
    return res;
}

int Solver1D::advanceTo(double tEnd, int maxSteps) {
    int n = 0;
    while (t_ < tEnd && n < maxSteps) {
        double dt = computeTimeStep();
        if (t_ + dt > tEnd) dt = tEnd - t_;
        if (dt <= 0.0) break;
        const StepResult r = step(dt);
        // Un pas abandonné n'est PAS un pas : ni compté, ni suivi d'un autre.
        // L'état et le temps sont restés ceux du début du pas (rollback).
        if (!r.ok) break;
        ++n;
    }
    return n;
}

void Solver1D::globalSums(double& sumD, double sumS[3], double& sumTau) const {
    sumD = 0.0; sumTau = 0.0;
    sumS[0] = sumS[1] = sumS[2] = 0.0;
    for (int i = NG; i < NG + cfg_.nx; ++i) {
        sumD   += cons_[i].D;
        sumTau += cons_[i].tau;
        for (int k = 0; k < 3; ++k) sumS[k] += cons_[i].S[k];
    }
    sumD *= dx_; sumTau *= dx_;
    for (int k = 0; k < 3; ++k) sumS[k] *= dx_;
}

}  // namespace grmhd
