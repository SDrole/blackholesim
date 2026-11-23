#pragma once

namespace grmhd {

//

struct RiemannState {
    double rho = 0.0;
    double p   = 0.0;
    double v   = 0.0;
};

enum class RiemannFailure {
    None = 0,
    InvalidInput,
    VacuumFormed,
    NoBracket,
    NotConverged
};

const char* toString(RiemannFailure f);

struct ExactRiemannSolution {
    bool   ok      = false;
    RiemannFailure failure = RiemannFailure::None;
    double pStar   = 0.0;
    double vStar   = 0.0;
    double rhoStarL = 0.0;
    double rhoStarR = 0.0;
    int    iterations = 0;
    double residual = 0.0;
};

//

bool riemannVacuumForms(const RiemannState& L, const RiemannState& R, double gamma);

ExactRiemannSolution solveExactRiemannSRHD(const RiemannState& L,
                                           const RiemannState& R,
                                           double gamma);

RiemannState sampleExactRiemannSRHD(const ExactRiemannSolution& sol,
                                    const RiemannState& L, const RiemannState& R,
                                    double gamma, double xi);

struct ShockResult {
    bool ok = false;
    double rhoPost = 0.0, vPost = 0.0, shockSpeed = 0.0;
};
ShockResult shockBranch(const RiemannState& pre, double pPost, double gamma, int sign);

//
double rarefactionVelocityAnalytic(const RiemannState& pre, double pPost,
                                   double gamma, int sign);
double rarefactionVelocityRK4(const RiemannState& pre, double pPost,
                              double gamma, int sign, int N);
long long rarefactionClampHits();
void resetRarefactionClampHits();

}  // namespace grmhd
