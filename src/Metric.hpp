#pragma once

namespace bhsim {

class Kerr {
public:
    Kerr(double M = 1.0, double a = 0.0) : M_(M), a_(a) {}

    double mass() const { return M_; }
    double spin() const { return a_; }              // a (longueur)
    double spinStar() const { return a_ / M_; }     // a/M ∈ [0,1)

    double delta(double r) const { return r * r - 2.0 * M_ * r + a_ * a_; }
    double sigma(double r, double cosTheta) const { return r * r + a_ * a_ * cosTheta * cosTheta; }

    struct ZamoFrame {
        double sigma, delta, A;     // Σ, Δ, A
        double lapse, omega;        // α = √(ΣΔ/A) ,  ω = 2Mar/A
        double gphiphi;             // g_φφ = A sin²θ / Σ
        double sinTheta, cosTheta;
    };
    ZamoFrame zamo(double r, double theta) const;

    double horizonOuter() const;   // r+ = M + sqrt(M² - a²)
    double horizonInner() const;   // r- = M - sqrt(M² - a²)
    double ergosphereEquator() const { return 2.0 * M_; }
    double iscoPrograde() const;

    double orbitalOmega(double r) const;   // Ω = dφ/dt
    double orbitalUt(double r) const;      // u^t = dt/dτ

    double diskRedshift(double r, double Lz,
                        double alphaObs = 1.0, double omegaObs = 0.0) const;

    double pageThorneFlux(double r) const;

private:
    double M_;
    double a_;
};

}  // namespace bhsim
