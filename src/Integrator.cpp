#include "Integrator.hpp"
#include <algorithm>
#include <cmath>

namespace bhsim {

double proposeStepFactor(double errorNorm, bool& accept, const StepControl& ctl) {
    accept = (errorNorm <= 1.0);

    double factor;
    if (errorNorm <= 1e-12) {
        factor = 5.0;                                  // erreur négligeable : on accélère
    } else {
        // Ordre 5  ->  exposant -1/5 sur la norme d'erreur (cible = 1).
        factor = ctl.safety * std::pow(errorNorm, -0.2);
    }
    return std::clamp(factor, 0.2, 5.0);
}

}  // namespace bhsim
