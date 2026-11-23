#pragma once
#include <vector>
#include <string>
#include "Fluid.hpp"
#include "EOS.hpp"
#include "Flux.hpp"

namespace grmhd {

//

enum class Reconstruction {
    PiecewiseConstant,   // 1er ordre — indispensable au DIAGNOSTIC : si un tube
    PLM_MinMod,
    PLM_MC
};

enum class BoundaryCondition {
    Outflow,     // copie de la cellule de bord (gradient nul)
    Periodic
};

//
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
    long long reconstructionFallbacks = 0;
    long long conservativeResyncs = 0;
    double minRhoBeforeFloor = 1e300;
    double minUBeforeFloor   = 1e300;

    void reset() { *this = FloorCounters{}; }
    long long totalFloors() const { return rhoFloored + uFloored; }
};

//
struct RejectedAttemptTelemetry {
    long long rejectedSteps = 0;
    FloorCounters during{};
    void reset() { *this = RejectedAttemptTelemetry{}; }
};

//
enum class RecoveryFailurePolicy {
    Abort,
    FallbackToPrevious
};

struct StepResult {
    bool ok = true;
    RecoveryFailure failure = RecoveryFailure::None;
    int  failedCell = -1;
    int  failedStage = -1;
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

//
class Solver1D {
public:
    explicit Solver1D(const Solver1DConfig& cfg);

    static constexpr int NG = 2;

    int    nx()   const { return cfg_.nx; }
    int    total() const { return cfg_.nx + 2 * NG; }
    double dx()   const { return dx_; }
    double time() const { return t_; }
    const Solver1DConfig& config() const { return cfg_; }
    const FloorCounters&  counters() const { return counters_; }
    FloorCounters& mutableCounters() { return counters_; }
    const RejectedAttemptTelemetry& rejectedTelemetry() const { return rejected_; }

    bool ok() const { return ok_; }

    double xc(int i) const { return cfg_.xmin + (i - NG + 0.5) * dx_; }

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
        faultStage_ = 0;
        faultCell_  = -1;
        curStage_   = 0;
    }

    void injectConservedForTest(int i, const ConservedState& U) { cons_[i] = U; }

    void injectRecoveryFaultForTest(int stage, int cell) {
        faultStage_ = stage;
        faultCell_  = cell;
    }

    double computeTimeStep() const;

    StepResult step(double dt);

    int advanceTo(double tEnd, int maxSteps = 2000000);

    const std::vector<PrimitiveState>& primitives() const { return prim_; }
    const std::vector<ConservedState>& conserved()  const { return cons_; }
    const IdealGasEOS& eos() const { return eos_; }

    void globalSums(double& sumD, double sumS[3], double& sumTau) const;

private:
    void applyBoundaryPrimitives();
    void applyBoundaryConserved();

    bool recoverWithPolicy(int cell, const ConservedState& U, const PrimitiveState& fallback,
                           PrimitiveState& out, RecoveryStatus& st, bool& usedFallback);

    bool computeRHS(const std::vector<ConservedState>& cons,
                    std::vector<std::array<double, 5>>& rhs, StepResult& res);
    bool recoverPrimitives(std::vector<ConservedState>& cons, StepResult& res);

    Solver1DConfig cfg_;
    IdealGasEOS    eos_;
    double dx_ = 0.0;
    double t_  = 0.0;
    std::vector<PrimitiveState> prim_;
    std::vector<ConservedState> cons_;
    void recordRejectedAttempt(const FloorCounters& before);

    FloorCounters counters_;
    RejectedAttemptTelemetry rejected_;
    bool ok_ = true;

    // Injection de faute (tests uniquement) — cf. injectRecoveryFaultForTest.
    int faultStage_ = 0;
    int faultCell_  = -1;
    int curStage_   = 0;
};

}  // namespace grmhd
