#pragma once
#include <vector>
#include <string>
#include "Fluid.hpp"
#include "EOS.hpp"
#include "Flux.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// S3 — Solveur SRHD 1D conservatif en espace plat (Minkowski).
//
// Découpage volontaire en briques indépendamment remplaçables :
//   état/EOS (S2) | flux (Flux.hpp) | reconstruction | Riemann (Riemann.hpp)
//   | CFL | mise à jour conservative | floors | conditions aux limites
// Aucune de ces briques n'appelle directement une autre en dur : le solveur
// les orchestre. Changer de reconstruction ou de solveur de Riemann ne doit
// toucher qu'un enum.
// -----------------------------------------------------------------------------

enum class Reconstruction {
    PiecewiseConstant,   // 1er ordre — indispensable au DIAGNOSTIC : si un tube
                          // à choc échoue en PLM et passe en PCM, c'est la
                          // reconstruction (ou son limiteur) qui est en cause,
                          // pas le solveur de Riemann.
    PLM_MinMod,
    PLM_MC
};

enum class BoundaryCondition {
    Outflow,     // copie de la cellule de bord (gradient nul)
    Periodic
};

// --- Politique de floors : couche EXPLICITE et comptabilisée -----------------
// S2 refuse délibérément de réparer un état non physique. S3 est l'endroit où
// une politique de floors peut exister — mais elle doit être identifiable,
// mesurable, et JUSTIFIÉE. Un test qui ne passe que parce que des milliers de
// cellules ont été « réparées » n'est pas un test qui passe.
struct FloorPolicy {
    bool   enabled = true;
    double rhoMin  = 1e-13;
    double uMin    = 1e-15;
};

struct FloorCounters {
    long long rhoFloored = 0;
    long long uFloored   = 0;
    long long coldAtMachinePrecision = 0;
    long long recoveryFailures       = 0;
    // Retour local au 1er ordre quand l'extrapolation PLM produirait ρ ou u
    // négatif. C'est une décision de la brique RECONSTRUCTION, distincte des
    // floors physiques — d'où un compteur séparé. Affirmer « il vaut zéro »
    // n'a de valeur que si c'est instrumenté.
    long long reconstructionFallbacks = 0;
    // Nombre de fois où U a dû être resynchronisé depuis P (donc où le schéma
    // a cessé d'être strictement conservatif sur cette cellule).
    long long conservativeResyncs = 0;
    double minRhoBeforeFloor = 1e300;   // extrema AVANT correction
    double minUBeforeFloor   = 1e300;

    void reset() { *this = FloorCounters{}; }
    long long totalFloors() const { return rhoFloored + uFloored; }
};

// --- SÉMANTIQUE DES DIAGNOSTICS ----------------------------------------------
// Deux significations distinctes, délibérément NON mélangées :
//
//  * `counters()` — ÉVOLUTION ACCEPTÉE. Décrit exclusivement la trajectoire que
//    le solveur a réellement retenue. Ces compteurs sont photographiés au début
//    du pas et RESTAURÉS par le rollback, exactement comme U et P : après un pas
//    abandonné, l'état ET son diagnostic correspondent au même instant. C'est
//    la seule sémantique qui rende une assertion du type « 0 floor sur ce
//    problème » vérifiable : sinon un floor survenu dans une tentative jetée
//    contaminerait le bilan d'une trajectoire qui ne l'a jamais subi.
//
//  * `rejectedTelemetry()` — TENTATIVES REJETÉES. Jamais annulée. C'est
//    l'information de diagnostic (« pourquoi ce pas a-t-il échoué ? »), qui
//    serait justement perdue si on la restaurait avec l'état.
//
// Les deux sont disjointes par construction : un événement compte dans l'une ou
// dans l'autre, jamais dans les deux.
struct RejectedAttemptTelemetry {
    long long rejectedSteps = 0;      // nombre de pas abandonnés
    FloorCounters during{};           // événements survenus PENDANT ces pas
    void reset() { *this = RejectedAttemptTelemetry{}; }
};

// --- Contrat en cas d'échec de primitive recovery ----------------------------
// S2 a délibérément refusé de réparer ; S3 doit donc CHOISIR une politique et
// l'assumer. Deux options, chacune adaptée à un usage différent :
//
//  * Abort (DÉFAUT, et le bon choix pour la validation scientifique) :
//    un échec de recovery sur un problème SRHD propre signale un vrai
//    problème (schéma, CFL, condition initiale). Continuer sur une ancienne
//    primitive produirait un résultat d'apparence correcte mais faux, et
//    masquerait la cause. Le pas est marqué en échec et le test doit échouer.
//
//  * FallbackToPrevious : réparation NON CONSERVATIVE (on repart de la
//    primitive précédente et on resynchronise U). Comptée, jamais silencieuse.
//    Réservée à un usage futur en production GRMHD, où les zones de jet à
//    très forte magnétisation rendent quelques échecs inévitables et où
//    arrêter la simulation n'est pas une option — c'est d'ailleurs ce que
//    font les codes du domaine (floors + reprise), mais toujours en
//    comptabilisant les cellules concernées.
enum class RecoveryFailurePolicy {
    Abort,
    FallbackToPrevious
};

// Résultat d'un pas de temps. Retourné explicitement plutôt que signalé par un
// seul drapeau global : l'appelant doit pouvoir décider (arrêter, réessayer
// avec un dt plus petit, diagnostiquer) sans inspecter l'état interne.
struct StepResult {
    bool ok = true;
    RecoveryFailure failure = RecoveryFailure::None;
    int  failedCell = -1;       // indice AVEC ghosts, -1 si aucun
    int  failedStage = -1;      // 1 ou 2 (étage SSPRK2), -1 si aucun
};

// --- Configuration d'une simulation 1D ---------------------------------------
struct Solver1DConfig {
    int    nx      = 200;
    double xmin    = 0.0;
    double xmax    = 1.0;
    double cfl     = 0.4;
    double gamma   = 5.0 / 3.0;
    Reconstruction     recon = Reconstruction::PLM_MC;
    BoundaryCondition  bc    = BoundaryCondition::Outflow;
    FloorPolicy        floors{};
    RecoveryFailurePolicy recoveryPolicy = RecoveryFailurePolicy::Abort;
};

// -----------------------------------------------------------------------------
// Solveur. Stocke primitives ET conservées ; les conservées font foi (c'est
// elles qu'on fait évoluer), les primitives sont reconstruites à chaque étage.
// -----------------------------------------------------------------------------
class Solver1D {
public:
    explicit Solver1D(const Solver1DConfig& cfg);

    static constexpr int NG = 2;   // 2 ghosts suffisent pour PLM

    int    nx()   const { return cfg_.nx; }
    int    total() const { return cfg_.nx + 2 * NG; }
    double dx()   const { return dx_; }
    double time() const { return t_; }
    const Solver1DConfig& config() const { return cfg_; }
    // Diagnostics de l'ÉVOLUTION ACCEPTÉE (restaurés par le rollback).
    const FloorCounters&  counters() const { return counters_; }
    FloorCounters& mutableCounters() { return counters_; }
    // Diagnostics des TENTATIVES REJETÉES (jamais restaurés).
    const RejectedAttemptTelemetry& rejectedTelemetry() const { return rejected_; }

    // Faux dès qu'un échec de recovery est survenu sous la politique Abort.
    // Un test scientifique DOIT le vérifier plutôt que de lire un résultat
    // produit par une réparation.
    bool ok() const { return ok_; }

    // Centre de la cellule i (indice AVEC ghosts).
    double xc(int i) const { return cfg_.xmin + (i - NG + 0.5) * dx_; }

    // Initialisation à partir d'une fonction x -> primitives.
    // Une réinitialisation doit rendre l'objet RÉELLEMENT neuf : temps,
    // diagnostic d'échec ET compteurs. Sinon un solveur ayant échoué
    // conserverait `ok_ == false` et des compteurs pollués, et le test suivant
    // hériterait silencieusement de l'état du précédent.
    template <typename Fn>
    void initialize(Fn&& f) {
        for (int i = 0; i < total(); ++i) prim_[i] = f(xc(i));
        applyBoundaryPrimitives();
        for (int i = 0; i < total(); ++i)
            cons_[i] = primitiveToConserved(prim_[i], eos_);
        t_ = 0.0;
        ok_ = true;
        counters_.reset();
        rejected_.reset();
        // Les coutures de TEST font partie de l'etat : « simulation neuve »
        // doit etre litteralement vrai, sinon une faute injectee survivrait a
        // la reinitialisation et contaminerait le scenario suivant.
        faultStage_ = 0;
        faultCell_  = -1;
        curStage_   = 0;
    }

    // ⚠️ RÉSERVÉ AUX TESTS : injecte un état conservé arbitraire pour exercer
    // les chemins d'échec de recovery, qui sont sinon inatteignables sur des
    // problèmes sains. Aucun usage en production.
    void injectConservedForTest(int i, const ConservedState& U) { cons_[i] = U; }

    // ⚠️ RÉSERVÉ AUX TESTS : force un échec de recovery à un ÉTAGE choisi.
    //
    // Pourquoi cette couture est nécessaire. Un état conservé inadmissible
    // injecté avant le pas est rejeté dès l'étage 1, c'est-à-dire AVANT que
    // `cons_` n'ait été modifié : un test « l'état est intact après un Abort »
    // passe alors même si le rollback est entièrement supprimé — il ne teste
    // rien. Le rollback n'a d'effet OBSERVABLE que si l'échec survient à
    // l'étage 2, quand `cons_` porte déjà la mise à jour de l'étage 1. Or on
    // ne peut pas atteindre l'étage 2 par injection d'état, puisque l'étage 1
    // rejetterait cet état. D'où une injection de faute explicite, la seule
    // façon honnête de rendre le test discriminant.
    //
    // stage = 0 désactive ; cell est un indice AVEC ghosts.
    void injectRecoveryFaultForTest(int stage, int cell) {
        faultStage_ = stage;
        faultCell_  = cell;
    }

    // Pas de temps CFL fondé sur la vitesse de signal RÉELLEMENT calculée
    // (max |λ±| sur le domaine), pas sur une estimation empirique.
    double computeTimeStep() const;

    // Un pas SSPRK2 (Heun fort-stabilité) :
    //   U^(1) = U^n + dt L(U^n)
    //   U^{n+1} = ½U^n + ½(U^(1) + dt L(U^(1)))
    //
    // TRANSACTIONNEL sous la politique Abort : si une conversion U→P échoue à
    // un quelconque étage, aucun flux n'est calculé à partir d'un état
    // périmé, `U` et `P` sont RESTAURÉS tels qu'au début du pas, et `t_`
    // n'avance pas. Justification du rollback plutôt que d'un état
    // partiellement mis à jour :
    //   * cohérence — un état à moitié avancé avec un `t_` inchangé ne
    //     correspond à AUCUN temps physique ;
    //   * il rend possible la réponse standard à un pas échoué : réessayer
    //     avec un dt plus petit, ce qu'un état détruit interdirait ;
    //   * c'est la sémantique d'« Abort » : le pas a eu lieu, ou pas.
    StepResult step(double dt);

    // Avance jusqu'à tEnd (dernier pas tronqué pour tomber exactement dessus).
    // Renvoie le nombre de pas effectués.
    int advanceTo(double tEnd, int maxSteps = 2000000);

    const std::vector<PrimitiveState>& primitives() const { return prim_; }
    const std::vector<ConservedState>& conserved()  const { return cons_; }
    const IdealGasEOS& eos() const { return eos_; }

    // Sommes globales des conservées sur les cellules PHYSIQUES seulement.
    void globalSums(double& sumD, double sumS[3], double& sumTau) const;

private:
    void applyBoundaryPrimitives();
    void applyBoundaryConserved();

    // POINT UNIQUE d'application de la politique de recovery — toute
    // conversion U→P du solveur passe par ici, pour qu'il n'existe pas deux
    // variantes du contrat qui pourraient diverger.
    // Renvoie false si l'appelant doit ABANDONNER (Abort + échec).
    bool recoverWithPolicy(int cell, const ConservedState& U, const PrimitiveState& fallback,
                           PrimitiveState& out, RecoveryStatus& st, bool& usedFallback);

    // L(U) : -(F_{i+1/2} - F_{i-1/2})/dx, écrit dans rhs.
    // Renvoie false sous Abort si une conversion échoue : AUCUN flux n'est
    // alors calculé (a fortiori aucun depuis une primitive périmée).
    bool computeRHS(const std::vector<ConservedState>& cons,
                    std::vector<std::array<double, 5>>& rhs, StepResult& res);
    // Reconstruit primitives depuis conservées, applique floors, compte.
    bool recoverPrimitives(std::vector<ConservedState>& cons, StepResult& res);

    Solver1DConfig cfg_;
    IdealGasEOS    eos_;
    double dx_ = 0.0;
    double t_  = 0.0;
    std::vector<PrimitiveState> prim_;
    std::vector<ConservedState> cons_;
    // Transfère vers `rejected_` les événements survenus pendant une tentative
    // abandonnée, puis laisse l'appelant restaurer `counters_`.
    void recordRejectedAttempt(const FloorCounters& before);

    FloorCounters counters_;
    RejectedAttemptTelemetry rejected_;
    bool ok_ = true;

    // Injection de faute (tests uniquement) — cf. injectRecoveryFaultForTest.
    int faultStage_ = 0;
    int faultCell_  = -1;
    int curStage_   = 0;   // étage SSPRK2 en cours d'exécution
};

}  // namespace grmhd
