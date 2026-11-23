//
#include "KerrSchild.hpp"
#include "Coordinates.hpp"

#include "Metric.hpp"      // bhsim::Kerr        (oracle BL)
#include "Geodesic.hpp"    // bhsim::geodesicRHS (oracle BL)

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace grmhd;

namespace {

int g_failures = 0;

void report(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  [%s] %-52s  err = %.3e  (tol %.1e)\n",
                ok ? "OK  " : "FAIL", name, err, tol);
    if (!ok) ++g_failures;
}

//

using cplx = std::complex<double>;

void metricDerivComplexStep(double r, double theta, double M, double a,
                            double dgdr[4][4], double dgdth[4][4]) {
    const double h = 1e-200;

    {   // ∂/∂r : r -> r + ih
        cplx gc[4][4];
        const cplx rr(r, h);
        const cplx st(std::sin(theta), 0.0), ct(std::cos(theta), 0.0);
        metricFromTrig<cplx>(rr, st, ct, cplx(M, 0.0), cplx(a, 0.0), gc);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                dgdr[i][j] = gc[i][j].imag() / h;
    }
    {
        cplx gc[4][4];
        const cplx th(theta, h);
        const cplx st = std::sin(th), ct = std::cos(th);
        metricFromTrig<cplx>(cplx(r, 0.0), st, ct, cplx(M, 0.0), cplx(a, 0.0), gc);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                dgdth[i][j] = gc[i][j].imag() / h;
    }
}

void testDerivativesVsComplexStep() {
    double worstR = 0.0, worstT = 0.0;
    const double spins[]  = {0.0, 0.5, 0.9, 0.998};
    const double radii[]  = {1.2, 2.0, 5.0, 20.0, 200.0};
    const double thetas[] = {0.25, 0.8, kPi / 2.0, 2.3, kPi - 0.25};

    for (double a : spins) {
        KerrSchild ks(1.0, a);
        for (double r : radii) {
            for (double th : thetas) {
                double dg[4][4][4];
                ks.metricDerivatives(r, th, dg);

                double refR[4][4], refT[4][4];
                metricDerivComplexStep(r, th, 1.0, a, refR, refT);

                double scale = 0.0;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        scale = std::max(scale, std::max(std::fabs(refR[i][j]),
                                                          std::fabs(refT[i][j])));
                scale = std::max(scale, 1e-30);

                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        worstR = std::max(worstR, std::fabs(dg[1][i][j] - refR[i][j]) / scale);
                        worstT = std::max(worstT, std::fabs(dg[2][i][j] - refT[i][j]) / scale);
                    }
                }
            }
        }
    }
    report("A1a d_r g: finite differences vs complex step", worstR, 1e-8);
    report("A1b d_theta g: finite differences vs complex step", worstT, 1e-8);
}

void testStepSensitivity() {
    const double r = 4.0, th = 1.0, a = 0.9;
    double refR[4][4], refT[4][4];
    metricDerivComplexStep(r, th, 1.0, a, refR, refT);

    std::printf("       h-step sweep (d_r g derivative, spin 0.9, r=4, theta=1) :\n");
    double bestErr = 1e30; double bestH = 0.0;
    double errAtProduction = 0.0;
    for (double h = 1e-2; h > 1e-12; h *= 0.1) {
        double gp[4][4], gm[4][4];
        KerrSchild ks(1.0, a);
        ks.metricCovariant(r + h, th, gp);
        ks.metricCovariant(r - h, th, gm);
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                scale = std::max(scale, std::fabs(refR[i][j]));
        scale = std::max(scale, 1e-30);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                worst = std::max(worst, std::fabs((gp[i][j] - gm[i][j]) / (2.0 * h) - refR[i][j]) / scale);
        std::printf("         h = %.0e  ->  err = %.3e\n", h, worst);
        if (worst < bestErr) { bestErr = worst; bestH = h; }
        if (std::fabs(h - 1e-6) < 1e-18) errAtProduction = worst;
    }
    std::printf("       optimum empirique : h = %.0e (err %.2e) ; h de production = 1e-6\n",
                bestH, bestErr);
    report("A2 h de production dans le plateau du balayage",
           errAtProduction / std::max(bestErr, 1e-30), 1e3);
}

void testAxisProximity() {
    const double a = 0.9, r = 4.0;
    KerrSchild ks(1.0, a);

    double worstParity = 0.0;
    for (double th : {1e-7, 1e-5, 1e-3, 1e-2, 0.1}) {
        double gp[4][4], gm[4][4];
        ks.metricCovariant(r, th, gp);
        ks.metricCovariant(r, -th, gm);
        double scale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) scale = std::max(scale, std::fabs(gp[i][j]));
        scale = std::max(scale, 1e-30);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                worstParity = std::max(worstParity, std::fabs(gp[i][j] - gm[i][j]) / scale);
    }
    report("A3a g(-theta) == g(theta): no spurious axis reflection", worstParity, 1e-15);

    //
    std::printf("       d_th g pres de l'axe -- trois normalisations :\n");
    std::printf("         %-10s %-12s %-14s %-14s %s\n",
                "theta", "|err| abs", "rel/|d_th g|", "rel/max|g|", "max|d_th g_ref|");
    double worstAbs = 0.0, worstVsG = 0.0;
    for (double th : {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}) {
        double dg[4][4][4];
        ks.metricDerivatives(r, th, dg);
        double refR[4][4], refT[4][4];
        metricDerivComplexStep(r, th, 1.0, a, refR, refT);
        double gg[4][4];
        ks.metricCovariant(r, th, gg);

        double absErr = 0.0, refScale = 0.0, gScale = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                absErr   = std::max(absErr, std::fabs(dg[2][i][j] - refT[i][j]));
                refScale = std::max(refScale, std::fabs(refT[i][j]));
                gScale   = std::max(gScale, std::fabs(gg[i][j]));
            }
        const double relRef = absErr / std::max(refScale, 1e-300);
        const double relG   = absErr / std::max(gScale, 1e-300);
        std::printf("         %-10.0e %-12.3e %-14.3e %-14.3e %.3e\n",
                    th, absErr, relRef, relG, refScale);
        worstAbs = std::max(worstAbs, absErr);
        worstVsG = std::max(worstVsG, relG);
    }
    report("A3b absolute d_theta g error remains bounded to the axis", worstAbs, 1e-7);
    report("A3c error relative to max|g| (non-vanishing scale)", worstVsG, 1e-9);

    std::printf("       relative Christoffel error near the axis :\n");
    double worstGamma = 0.0;
    for (double th : {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5}) {
        Metric4 m;
        if (!ks.metricAt(r, th, m)) continue;

        double refR[4][4], refT[4][4];
        metricDerivComplexStep(r, th, 1.0, a, refR, refT);
        double dgRef[4][4][4] = {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) { dgRef[1][i][j] = refR[i][j]; dgRef[2][i][j] = refT[i][j]; }

        double Gref[4][4][4];
        for (int l = 0; l < 4; ++l)
            for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu) {
                    double s = 0.0;
                    for (int sg = 0; sg < 4; ++sg)
                        s += m.ginv[l][sg] * (dgRef[mu][sg][nu] + dgRef[nu][sg][mu] - dgRef[sg][mu][nu]);
                    Gref[l][mu][nu] = 0.5 * s;
                }

        double Gfd[4][4][4];
        if (!ks.christoffel(r, th, Gfd)) continue;

        double err = 0.0, scale = 0.0;
        for (int l = 0; l < 4; ++l)
            for (int mu = 0; mu < 4; ++mu)
                for (int nu = 0; nu < 4; ++nu) {
                    err   = std::max(err, std::fabs(Gfd[l][mu][nu] - Gref[l][mu][nu]));
                    scale = std::max(scale, std::fabs(Gref[l][mu][nu]));
                }
        const double rel = err / std::max(scale, 1e-300);
        std::printf("         theta = %-8.0e  err_rel(Gamma) = %.3e   (max|Gamma| = %.3e)\n",
                    th, rel, scale);
        worstGamma = std::max(worstGamma, rel);
    }
    report("A3d Christoffels: relative error bounded down to 1e-5 rad", worstGamma, 1e-8);
}

//

struct GeoIC {
    double a;          // spin
    double r0, th0;    // position initiale
    double pr, pth;    // composantes contravariantes p^r, p^θ
    double pphi;       // p^φ (en KS)
    const char* label;
};

struct KSResult { double r, th, pr, pth; double dE, dL, dNull; bool ok; };

KSResult integrateKS(const KerrSchild& ks, const GeoIC& ic, double dl, int nsteps) {
    KSResult out{};
    double x[4] = {0.0, ic.r0, ic.th0, 0.0};
    double p[4] = {0.0, ic.pr, ic.pth, ic.pphi};

    Metric4 m;
    if (!ks.metricAt(x[1], x[2], m)) { out.ok = false; return out; }
    const double A = m.g[0][0];
    double B = 0.0, C = 0.0;
    for (int i = 1; i < 4; ++i) B += m.g[0][i] * p[i];
    for (int i = 1; i < 4; ++i)
        for (int j = 1; j < 4; ++j) C += m.g[i][j] * p[i] * p[j];
    const double disc = B * B - A * C;
    if (disc < 0.0) { out.ok = false; return out; }
    p[0] = (-B - std::sqrt(disc)) / A;

    auto cov = [&](const double xx[4], const double pp[4], int lo) {
        Metric4 mm; ks.metricAt(xx[1], xx[2], mm);
        double v = 0.0; for (int i = 0; i < 4; ++i) v += mm.g[lo][i] * pp[i];
        return v;
    };
    auto nrm = [&](const double xx[4], const double pp[4]) {
        Metric4 mm; ks.metricAt(xx[1], xx[2], mm);
        double v = 0.0;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) v += mm.g[i][j] * pp[i] * pp[j];
        return v;
    };

    const double E0 = cov(x, p, 0), L0 = cov(x, p, 3);
    double pScale = 0.0;
    for (int i = 0; i < 4; ++i) pScale = std::max(pScale, std::fabs(p[i]));
    const double nullScale = std::max(pScale * pScale, 1e-30);

    bool gammaOk = true;
    auto rhs = [&](const double xx[4], const double pp[4], double dx[4], double dp[4]) {
        double G[4][4][4];
        if (!ks.christoffel(xx[1], xx[2], G)) gammaOk = false;
        for (int i = 0; i < 4; ++i) dx[i] = pp[i];
        for (int l = 0; l < 4; ++l) {
            double acc = 0.0;
            for (int aa = 0; aa < 4; ++aa)
                for (int bb = 0; bb < 4; ++bb)
                    acc -= G[l][aa][bb] * pp[aa] * pp[bb];
            dp[l] = acc;
        }
    };

    for (int s = 0; s < nsteps; ++s) {
        double k1x[4],k1p[4],k2x[4],k2p[4],k3x[4],k3p[4],k4x[4],k4p[4],tx[4],tp[4];
        rhs(x,p,k1x,k1p);
        for (int i=0;i<4;++i){tx[i]=x[i]+0.5*dl*k1x[i];tp[i]=p[i]+0.5*dl*k1p[i];}
        rhs(tx,tp,k2x,k2p);
        for (int i=0;i<4;++i){tx[i]=x[i]+0.5*dl*k2x[i];tp[i]=p[i]+0.5*dl*k2p[i];}
        rhs(tx,tp,k3x,k3p);
        for (int i=0;i<4;++i){tx[i]=x[i]+dl*k3x[i];tp[i]=p[i]+dl*k3p[i];}
        rhs(tx,tp,k4x,k4p);
        for (int i=0;i<4;++i){
            x[i]+=(dl/6.0)*(k1x[i]+2.0*k2x[i]+2.0*k3x[i]+k4x[i]);
            p[i]+=(dl/6.0)*(k1p[i]+2.0*k2p[i]+2.0*k3p[i]+k4p[i]);
        }
        out.dE = std::max(out.dE, std::fabs(cov(x,p,0)-E0)/std::max(std::fabs(E0),1e-30));
        out.dL = std::max(out.dL, std::fabs(cov(x,p,3)-L0)/std::max(std::fabs(L0),1e-30));
        out.dNull = std::max(out.dNull, std::fabs(nrm(x,p))/nullScale);
        if (!gammaOk) { out.ok = false; return out; }
    }
    out.r = x[1]; out.th = x[2]; out.pr = p[1]; out.pth = p[2];
    out.ok = true;
    return out;
}

struct BLResult { double r, th, pr, pth; double dK; bool ok; };

BLResult integrateBL(const GeoIC& ic, double dl, int nsteps) {
    BLResult out{};
    const double M = 1.0, a = ic.a;
    const bhsim::Kerr bh(M, a);

    const double r = ic.r0, th = ic.th0;
    const double Delta = r * r - 2.0 * M * r + a * a;
    if (std::fabs(Delta) < 1e-12) { out.ok = false; return out; }

    // p^φ_BL = p^φ_KS - (a/Δ) p^r   (p^r, p^θ identiques)
    const double pphiBL = ic.pphi - (a / Delta) * ic.pr;

    const double st = std::sin(th), ct = std::cos(th);
    const double Sigma = r * r + a * a * ct * ct;
    const double s2 = st * st;
    const double gtt = -(1.0 - 2.0 * M * r / Sigma);
    const double gtp = -2.0 * M * a * r * s2 / Sigma;
    const double grr = Sigma / Delta;
    const double gthth = Sigma;
    const double gpp = s2 * (r * r + a * a + 2.0 * M * a * a * r * s2 / Sigma);

    const double A = gtt;
    const double B = gtp * pphiBL;
    const double C = grr * ic.pr * ic.pr + gthth * ic.pth * ic.pth + gpp * pphiBL * pphiBL;
    const double disc = B * B - A * C;
    if (disc < 0.0) { out.ok = false; return out; }
    const double ptBL = (-B - std::sqrt(disc)) / A;

    // Constantes du mouvement (covariantes) attendues par le code bhsim.
    const double p_t   = gtt * ptBL + gtp * pphiBL;
    const double p_phi = gtp * ptBL + gpp * pphiBL;
    const double p_r   = grr * ic.pr;
    const double p_th  = gthth * ic.pth;

    bhsim::Constants c;
    c.E  = -p_t;
    c.Lz = p_phi;
    c.Q  = p_th * p_th + ct * ct * (c.Lz * c.Lz / std::max(s2, 1e-300) - a * a * c.E * c.E);
    const double LmaE = c.Lz - a * c.E;
    c.K = c.Q + LmaE * LmaE;

    bhsim::State y;
    y.r = r; y.theta = th; y.phi = 0.0; y.pr = p_r; y.pth = p_th;

    const double K0 = c.K;
    auto f = [&](const bhsim::State& s) { return bhsim::geodesicRHS(s, bh, c); };

    for (int s = 0; s < nsteps; ++s) {
        const bhsim::State k1 = f(y);
        bhsim::State t2 = y + (0.5 * dl) * k1; const bhsim::State k2 = f(t2);
        bhsim::State t3 = y + (0.5 * dl) * k2; const bhsim::State k3 = f(t3);
        bhsim::State t4 = y + dl * k3;         const bhsim::State k4 = f(t4);
        y = y + (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        const double drift = std::fabs(bhsim::carterK(y, bh, c) - K0) / (std::fabs(K0) + 1e-30);
        out.dK = std::max(out.dK, drift);
    }

    const double ctF = std::cos(y.theta), stF = std::sin(y.theta);
    const double SigmaF = y.r * y.r + a * a * ctF * ctF;
    const double DeltaF = y.r * y.r - 2.0 * M * y.r + a * a;
    out.r = y.r; out.th = y.theta;
    out.pr  = (DeltaF / SigmaF) * y.pr;    // p^r = g^rr p_r  en BL
    out.pth = y.pth / SigmaF;              // p^θ = g^θθ p_θ
    (void)stF;
    out.ok = true;
    return out;
}

// B1 — comparaison directe + B2 convergence.
void testCrossValidationKSvsBL() {
    const GeoIC ics[] = {
        {0.0,  20.0, kPi / 2.0, -0.35,  0.00,  0.010, "a=0    equatorial"},
        {0.5,  20.0, 1.10,      -0.35,  0.03,  0.008, "a=0.5  incline"},
        {0.9,  15.0, kPi / 2.0, -0.30,  0.00,  0.012, "a=0.9  equatorial"},
        {0.9,  12.0, 0.90,      -0.40,  0.04,  0.006, "a=0.9  incline"},
        {0.998, 8.0, kPi / 2.0, -0.25,  0.00,  0.015, "a=0.998 near horizon"},
    };

    std::printf("       comparaison KS (Christoffels) vs BL (hamiltonien bhsim) :\n");
    double worstDiff = 0.0, worstCarter = 0.0, worstKSinv = 0.0;

    for (const GeoIC& ic : ics) {
        KerrSchild ks(1.0, ic.a);
        const double dl = 0.01;
        const int n = 2000;   // λ_final = 20

        const KSResult rk = integrateKS(ks, ic, dl, n);
        const BLResult rb = integrateBL(ic, dl, n);
        if (!rk.ok || !rb.ok) { worstDiff = 1e30; continue; }

        const double er  = std::fabs(rk.r - rb.r) / std::max(std::fabs(rb.r), 1e-30);
        const double eth = std::fabs(rk.th - rb.th);
        const double epr = std::fabs(rk.pr - rb.pr) / std::max(std::fabs(rb.pr), 1e-30);

        std::printf("         %-24s r: %.3e  th: %.3e  p^r: %.3e  | Carter(BL) %.2e\n",
                    ic.label, er, eth, epr, rb.dK);
        worstDiff = std::max(worstDiff, std::max(er, eth));
        worstCarter = std::max(worstCarter, rb.dK);
        worstKSinv = std::max(worstKSinv, std::max(rk.dE, std::max(rk.dL, rk.dNull)));
    }

    report("B1a accord KS vs BL sur (r, theta) a lambda=20", worstDiff, 1e-6);
    report("B1b invariants KS (E, Lz, nullite) le long du trajet", worstKSinv, 1e-8);
    report("B1c Carter conserved on BL side (oracle sanity check)", worstCarter, 1e-8);
}

void testConvergenceOrder() {
    const GeoIC ic = {0.9, 15.0, 1.05, -0.30, 0.03, 0.010, "convergence"};
    KerrSchild ks(1.0, ic.a);

    constexpr double kFloor = 1e-11;

    std::printf("       convergence de l'ecart KS<->BL (lambda_final = 10 fixe) :\n");
    double prevErr = 0.0, prevDl = 0.0;
    double bestOrder = 0.0;
    bool measuredInTruncationRegime = false;

    for (int k = 0; k < 7; ++k) {
        const double dl = 0.8 / std::pow(2.0, k);
        const int n = static_cast<int>(std::lround(10.0 / dl));
        const KSResult rk = integrateKS(ks, ic, dl, n);
        const BLResult rb = integrateBL(ic, dl, n);
        if (!rk.ok || !rb.ok) continue;
        const double e = std::fabs(rk.r - rb.r) + std::fabs(rk.th - rb.th);

        std::printf("         dl = %.5f  ecart = %.3e", dl, e);
        if (k > 0 && prevErr > 0.0 && e > 0.0) {
            const double order = std::log2(prevErr / e);
            const bool bothAboveFloor = (e > 100.0 * kFloor) && (prevErr > 100.0 * kFloor);
            std::printf("   ordre = %+.2f %s", order,
                        bothAboveFloor ? "(regime troncature)" : "(plancher atteint)");
            if (bothAboveFloor) {
                bestOrder = std::max(bestOrder, order);
                measuredInTruncationRegime = true;
            }
        }
        std::printf("\n");
        prevErr = e; prevDl = dl;
    }
    (void)prevDl;

    report("B2a mesure effectuee dans le regime de troncature",
           measuredInTruncationRegime ? 0.0 : 1.0, 1e-15);
    report("B2b ordre de convergence observe >= 3",
           (bestOrder >= 3.0) ? 0.0 : 1.0, 1e-15);
    std::printf("       (ordre max observe hors plancher : %.2f ; le plateau final a\n"
                "        ~1e-12 vient du plancher de differentiation des Christoffels,\n"
                "        this is expected and not a failure)\n", bestOrder);
}

}  // namespace

int main() {
    std::printf("=====================================================================\n");
    std::printf("  S1 CONSOLIDATION : schema de derivation + croisement KS <-> BL\n");
    std::printf("=====================================================================\n");
    std::printf("\n  BLOC A -- verification du schema de differentiation\n");
    testDerivativesVsComplexStep();
    testStepSensitivity();
    testAxisProximity();

    std::printf("\n  BLOC B -- validation croisee contre l'oracle Boyer-Lindquist\n");
    testCrossValidationKSvsBL();
    testConvergenceOrder();

    std::printf("---------------------------------------------------------------------\n");
    if (g_failures == 0)
        std::printf("  CONSOLIDATION S1 VALIDEE -- la geometrie est verrouillee, S2 peut demarrer.\n");
    else
        std::printf("  %d FAILURE(S) -- geometry is not locked; do not start S2.\n", g_failures);
    std::printf("=====================================================================\n");
    return g_failures == 0 ? 0 : 1;
}
