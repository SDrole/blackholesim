#pragma once
#include "EOS.hpp"

namespace grmhd {

//

struct PrimitiveState {
    double rho = 0.0;
    double u   = 0.0;
    double utilde[3] = {0.0, 0.0, 0.0};
};

struct ConservedState {
    double D   = 0.0;        // ρW
    double S[3] = {0.0, 0.0, 0.0};        // ρhW²v_i
    double tau = 0.0;        // ρhW² - p - D
};

//
enum class RecoveryFailure {
    None = 0,
    NonFiniteInput,
    NonPositiveDensity,    // D <= 0
    EnergyBelowMomentum,
    BracketFailed,
    NoConvergence,
    UnphysicalResult,
    NegativeInternalEnergy
};

struct RecoveryStatus {
    bool converged = false;
    int  iterations = 0;
    double residual = 0.0;             // |f(p)| final
    RecoveryFailure failure = RecoveryFailure::None;

    double lorentzFactor = 0.0;
    bool   usedBisection = false;

    bool coldAtMachinePrecision = false;
};

const char* toString(RecoveryFailure f);

// --- Conversions -------------------------------------------------------------

ConservedState primitiveToConserved(const PrimitiveState& P, const IdealGasEOS& eos);

bool conservedToPrimitive(const ConservedState& U, const IdealGasEOS& eos,
                          PrimitiveState& P, RecoveryStatus& status);

double lorentzFactor(const PrimitiveState& P);

}  // namespace grmhd
