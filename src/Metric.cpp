#include "Metric.hpp"
#include "Config.hpp"   // kPi
#include <algorithm>
#include <cmath>

namespace bhsim {

double Kerr::horizonOuter() const {
    const double disc = M_ * M_ - a_ * a_;
    return M_ + std::sqrt(disc > 0.0 ? disc : 0.0);
}

double Kerr::horizonInner() const {
    const double disc = M_ * M_ - a_ * a_;
    return M_ - std::sqrt(disc > 0.0 ? disc : 0.0);
}

Kerr::ZamoFrame Kerr::zamo(double r, double theta) const {
    const double s = std::sin(theta), c = std::cos(theta);
    const double Sig  = r * r + a_ * a_ * c * c;
    const double Del  = r * r - 2.0 * M_ * r + a_ * a_;
    const double r2a2 = r * r + a_ * a_;
    const double A    = r2a2 * r2a2 - a_ * a_ * Del * s * s;   // (r²+a²)² - a²Δsin²θ

    ZamoFrame z;
    z.sigma    = Sig;
    z.delta    = Del;
    z.A        = A;
    z.lapse    = std::sqrt(Sig * Del / A);     // α = √(ΣΔ/A)
    z.omega    = 2.0 * M_ * a_ * r / A;        // ω = 2Mar/A = -g_tφ/g_φφ
    z.gphiphi  = A * s * s / Sig;              // g_φφ = A sin²θ / Σ
    z.sinTheta = s;
    z.cosTheta = c;
    return z;
}

double Kerr::iscoPrograde() const {
    // Formule de Bardeen, Press & Teukolsky (1972).
    const double as = a_ / M_;
    const double a2 = as * as;
    const double Z1 = 1.0 + std::cbrt(1.0 - a2) * (std::cbrt(1.0 + as) + std::cbrt(1.0 - as));
    const double Z2 = std::sqrt(3.0 * a2 + Z1 * Z1);
    // Signe '-' : orbite prograde (co-rotative).
    return M_ * (3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

double Kerr::orbitalOmega(double r) const {
    // Ω = √M / (r^{3/2} + a√M)   (prograde)
    return std::sqrt(M_) / (std::pow(r, 1.5) + a_ * std::sqrt(M_));
}

double Kerr::orbitalUt(double r) const {
    // u^t = (r^{3/2} + a√M) / √(r³ - 3M r² + 2a√M r^{3/2})
    const double num = std::pow(r, 1.5) + a_ * std::sqrt(M_);
    const double rad = r * r * r - 3.0 * M_ * r * r + 2.0 * a_ * std::sqrt(M_) * std::pow(r, 1.5);
    return num / std::sqrt(rad > 0.0 ? rad : 1e-12);
}

double Kerr::diskRedshift(double r, double Lz, double alphaObs, double omegaObs) const {
    const double ut = orbitalUt(r);
    const double denom = alphaObs * ut * (1.0 - orbitalOmega(r) * Lz);
    if (denom <= 1e-6) return 0.0;     // garde-fou (faisceau quasi tangent)
    return (1.0 - omegaObs * Lz) / denom;
}

double Kerr::pageThorneFlux(double r) const {
    const double rin = iscoPrograde();
    if (r <= rin) return 0.0;

    const double as = a_ / M_;
    const double x  = std::sqrt(r / M_);
    const double x0 = std::sqrt(rin / M_);

    // Racines de x³ - 3x + 2a* = 0.
    const double t  = std::acos(std::clamp(as, -1.0, 1.0)) / 3.0;
    const double third = kPi / 3.0;
    const double x1 =  2.0 * std::cos(t - third);
    const double x2 =  2.0 * std::cos(t + third);
    const double x3 = -2.0 * std::cos(t);

    auto coeff = [&](double xi, double xj, double xk) {
        if (std::fabs(xi) < 1e-9) return 0.0;
        return 3.0 * (xi - as) * (xi - as) / (xi * (xi - xj) * (xi - xk));
    };
    const double c1 = coeff(x1, x2, x3);
    const double c2 = coeff(x2, x1, x3);
    const double c3 = coeff(x3, x1, x2);

    const double bracket =
        x - x0 - 1.5 * as * std::log(x / x0)
        - c1 * std::log((x - x1) / (x0 - x1))
        - c2 * std::log((x - x2) / (x0 - x2))
        - c3 * std::log((x - x3) / (x0 - x3));

    const double denom = x * x * x * x * (x * x * x - 3.0 * x + 2.0 * as);
    const double f = bracket / denom;
    return f > 0.0 ? f : 0.0;
}

}  // namespace bhsim
