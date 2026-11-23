#pragma once

namespace grmhd {

// -----------------------------------------------------------------------------
// Kerr metric in spherical KERR-SCHILD coordinates (t, r, theta, phi).
//
// WHY Kerr-Schild instead of Boyer-Lindquist (used by the ray tracer):
// BL is singular at the horizon (g_rr -> infinity as Delta -> 0), which makes
// it unsuitable for evolving fluid through the horizon. Kerr-Schild remains
// regular everywhere except the physical ring singularity. This is the
// standard choice in GRMHD codes such as HARM, iharm3D, Athena++ and BHAC,
// which place their inner boundary inside the horizon.
//
// Signature convention: (-,+,+,+). Geometrized units G=c=1.
// Indices: 0=t, 1=r, 2=theta, 3=phi.
//
// Sigma = r^2 + a^2 cos^2(theta), Delta = r^2 - 2Mr + a^2
//
// Non-zero covariant components:
//   g_tt   = -(1 - 2Mr/Sigma)          g_tr   = 2Mr/Sigma
//   g_tphi = -2Mar sin^2(theta)/Sigma  g_rr   = 1 + 2Mr/Sigma
//   g_rphi = -a sin^2(theta)(1+2Mr/Sigma)
//   g_thetatheta = Sigma
//   g_phiphi = sin^2(theta)[Sigma + a^2 sin^2(theta)(1+2Mr/Sigma)]
//
// Exact analytic result used as an independent check: sqrt(-g)=Sigma sin(theta).
// -----------------------------------------------------------------------------

struct Metric4 {
    double g[4][4]    = {};   // covariant g_mu_nu
    double ginv[4][4] = {};   // contravariant g^mu_nu
    double sqrtNegG   = 0.0;  // sqrt(-det g)
};

// Analytic metric kernel templated on scalar type and parameterized by
// (sin(theta), cos(theta)) rather than theta itself.
//
// This keeps the polar-axis floor outside the analytic core and allows the
// function to be instantiated with std::complex for complex-step
// differentiation: f'(x) ~= Im[f(x+ih)]/h. Because it avoids subtraction,
// complex-step differentiation provides an independent near-machine-precision
// reference for validating the finite-difference Christoffel derivatives.
template <typename T>
void metricFromTrig(const T& r, const T& st, const T& ct,
                    const T& M, const T& a, T g[4][4]) {
    const T s2 = st * st;
    const T Sigma = r * r + a * a * ct * ct;
    const T twoMrOverSigma = T(2.0) * M * r / Sigma;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            g[i][j] = T(0.0);

    g[0][0] = -(T(1.0) - twoMrOverSigma);
    g[0][1] = twoMrOverSigma;
    g[1][0] = g[0][1];
    g[0][3] = -twoMrOverSigma * a * s2;
    g[3][0] = g[0][3];

    g[1][1] = T(1.0) + twoMrOverSigma;
    g[1][3] = -a * s2 * (T(1.0) + twoMrOverSigma);
    g[3][1] = g[1][3];

    g[2][2] = Sigma;

    g[3][3] = s2 * (Sigma + a * a * s2 * (T(1.0) + twoMrOverSigma));
}

class KerrSchild {
public:
    KerrSchild(double M = 1.0, double a = 0.0) : M_(M), a_(a) {}

    double mass() const { return M_; }
    double spin() const { return a_; }

    // Supported Kerr domain: M>0 and |a|<=M. The extremal case |a|=M is
    // valid; |a|>M is a naked singularity and is rejected.
    bool parametersValid() const;

    // Horizon radii r+/- = M +/- sqrt(M^2-a^2).
    // Returns false outside the supported Kerr domain.
    bool horizons(double& rPlus, double& rMinus) const;

    // Convenience accessors. Outside the supported domain they return NaN so
    // an invalid configuration propagates visibly instead of becoming a
    // plausible horizon value.
    double horizonOuter() const;
    double horizonInner() const;

    // Analytic covariant metric.
    void metricCovariant(double r, double theta, double g[4][4]) const;

    // Full metric: covariant + numerical inverse + sqrt(-g).
    // Gauss-Jordan inversion is used instead of a hand-transcribed inverse to
    // reduce algebraic transcription risk; g*g^-1=I is then a real test of the
    // inversion routine.
    bool metricAt(double r, double theta, Metric4& out) const;

    // Analytic sqrt(-g)=Sigma sin(theta), used as an independent reference.
    double sqrtNegGAnalytic(double r, double theta) const;

    // Christoffel symbols Gamma^lambda_mu_nu from numerical derivatives of the
    // metric. This is less error-prone than transcribing the full analytic
    // expressions and its truncation error is well below the hydro scheme's
    // error scale.
    //
    // Kerr in spherical KS is stationary and axisymmetric:
    // partial_t g = partial_phi g = 0, so only r and theta derivatives remain.
    //
    // Returns false if the metric cannot be inverted. Gamma is zero-filled on
    // failure only to avoid uninitialized output; callers MUST check the
    // return value. Silent Gamma=0 would mimic flat space and become dangerous
    // once S4 source terms depend on the connection.
    [[nodiscard]] bool christoffel(double r, double theta, double Gamma[4][4][4]) const;

    // Numerical partial_lambda g_mu_nu, exposed for validation.
    void metricDerivatives(double r, double theta, double dg[4][4][4]) const;

private:
    double M_;
    double a_;
};

}  // namespace grmhd
