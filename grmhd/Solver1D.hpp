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
    double minRhoBeforeFloor = 1e300;   // extrema AVANT correction
    double minUBeforeFloor   = 1e300;

    void reset() { *this = FloorCounters{}; }
    long long totalFloors() const { return rhoFloored + uFloored; }
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
    const FloorCounters&  counters() const { return counters_; }
    FloorCounters& mutableCounters() { return counters_; }

    // Centre de la cellule i (indice AVEC ghosts).
    double xc(int i) const { return cfg_.xmin + (i - NG + 0.5) * dx_; }

    // Initialisation à partir d'une fonction x -> primitives.
    template <typename Fn>
    void initialize(Fn&& f) {
        for (int i = 0; i < total(); ++i) prim_[i] = f(xc(i));
        applyBoundaryPrimitives();
        for (int i = 0; i < total(); ++i)
            cons_[i] = primitiveToConserved(prim_[i], eos_);
        t_ = 0.0;
    }

    // Pas de temps CFL fondé sur la vitesse de signal RÉELLEMENT calculée
    // (max |λ±| sur le domaine), pas sur une estimation empirique.
    double computeTimeStep() const;

    // Un pas SSPRK2 (Heun fort-stabilité) :
    //   U^(1) = U^n + dt L(U^n)
    //   U^{n+1} = ½U^n + ½(U^(1) + dt L(U^(1)))
    void step(double dt);

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
    // L(U) : -(F_{i+1/2} - F_{i-1/2})/dx, écrit dans rhs.
    void computeRHS(const std::vector<ConservedState>& cons,
                    std::vector<std::array<double, 5>>& rhs);
    // Reconstruit primitives depuis conservées, applique floors, compte.
    void recoverPrimitives(std::vector<ConservedState>& cons);

    Solver1DConfig cfg_;
    IdealGasEOS    eos_;
    double dx_ = 0.0;
    double t_  = 0.0;
    std::vector<PrimitiveState> prim_;
    std::vector<ConservedState> cons_;
    FloorCounters counters_;
};

}  // namespace grmhd
