#pragma once
#include "Geodesic.hpp"

namespace bhsim {


struct StepControl {
    double hMin = 1e-6;
    double hMax = 50.0;
    double safety = 0.9;
};

double proposeStepFactor(double errorNorm, bool& accept, const StepControl& ctl);

// --- Runge-Kutta 4 classique (pas fixe) -------------------------------------
template <typename Deriv>
State rk4Step(const State& y, double h, Deriv&& f) {
    const State k1 = f(y);
    const State k2 = f(y + (0.5 * h) * k1);
    const State k3 = f(y + (0.5 * h) * k2);
    const State k4 = f(y + h * k3);
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// --- Runge-Kutta-Fehlberg 4(5) (pas adaptatif) ------------------------------
struct RKF45Result {
    State y;
    State err;
};

template <typename Deriv>
RKF45Result rkf45Step(const State& y, double h, Deriv&& f) {
    const State k1 = f(y);
    const State k2 = f(y + h * (0.25 * k1));
    const State k3 = f(y + h * ((3.0 / 32.0) * k1 + (9.0 / 32.0) * k2));
    const State k4 = f(y + h * ((1932.0 / 2197.0) * k1 - (7200.0 / 2197.0) * k2
                              + (7296.0 / 2197.0) * k3));
    const State k5 = f(y + h * ((439.0 / 216.0) * k1 - 8.0 * k2
                              + (3680.0 / 513.0) * k3 - (845.0 / 4104.0) * k4));
    const State k6 = f(y + h * ((-8.0 / 27.0) * k1 + 2.0 * k2
                              - (3544.0 / 2565.0) * k3 + (1859.0 / 4104.0) * k4
                              - (11.0 / 40.0) * k5));

    const State y4 = y + h * ((25.0 / 216.0) * k1 + (1408.0 / 2565.0) * k3
                           + (2197.0 / 4104.0) * k4 - (1.0 / 5.0) * k5);
    const State y5 = y + h * ((16.0 / 135.0) * k1 + (6656.0 / 12825.0) * k3
                           + (28561.0 / 56430.0) * k4 - (9.0 / 50.0) * k5
                           + (2.0 / 55.0) * k6);

    return { y5, y5 - y4 };
}

}  // namespace bhsim
