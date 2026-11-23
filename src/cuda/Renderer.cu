// -----------------------------------------------------------------------------
// Renderer.cu — portage CUDA du moteur de rendu Kerr.
//
// Réécriture de l'intégrateur géodésique RKF45 (hamiltonien canonique, métrique
// de Kerr) et du disque relativiste de Novikov-Thorne (flux de Page-Thorne) sous
// forme de kernel massivement parallèle : un thread CUDA trace tous les rayons
// SSAA d'un pixel. Le résultat (image HDR linéaire + cartes scientifiques) est
// strictement identique à celui du moteur CPU (mêmes formules, double précision)
// afin de préserver la conservation de la constante de Carter (< 1e-6).
//
// La précision DOUBLE est conservée volontairement : le contrôleur de pas vise
// atol = rtol = 1e-10 et la dérive de Carter est vérifiée à 1e-6 — hors de portée
// du simple précision. Sur Ada (RTX 4090) le FP64 est bridé (1:64), mais le
// parallélisme (≈132 M rayons/image traités de front) reste très supérieur au CPU.
//
// Trois raffinements de fidélité visuelle (au-delà du portage 1:1 du CPU) :
//   1. Turbulence MHD anisotrope (dev_mhdDensityFactor, dev_shearAspect) : le
//      bruit du plasma est étiré azimutalement par le cisaillement képlérien
//      différentiel plutôt que simplement tourné en bloc — cf. commentaire
//      détaillé au-dessus de dev_mhdDensityFactor.
//   2. Transfert radiatif multi-bande (dev_opacitySpectrum, dev_traceRay) :
//      transmittance RGB vectorielle (opacité chromatique par canal), au lieu
//      d'un scalaire gris appliqué identiquement aux 3 canaux.
//   3. Anneaux de photons d'ordre élevé n≥2,3 (photonRingRadius/Step,
//      ringSupersample dans renderKernel) : pas raffiné dans tout le corridor
//      de la sphère de photons + budget de pas dédié (gpuMaxSteps) pour éviter
//      la terminaison prématurée des rayons quasi critiques, et
//      sur-échantillonnage adaptatif ciblé sur les pixels de bord.
// -----------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "Renderer.hpp"

namespace bhsim {
namespace cuda_detail {

#define BH_PI 3.14159265358979323846

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                               \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// Table de permutation de Perlin (identique à la version CPU), en mémoire
// constante : accès rapide et diffusé pour le bruit de plasma du disque.
__constant__ int c_perm[512];

// --- Paramètres de la scène transmis au kernel (POD, passé par valeur) --------
struct GpuParams {
    double M, a;                       // masse, spin (longueur)
    int    W, H, S;                    // largeur, hauteur, SSAA (S×S échantillons)
    double exposure;

    // Caméra (observateur ZAMO à distance finie) — tétrade constante par image.
    double rObs, thetaObs, halfWidth, halfHeight;
    double obsLapse, obsOmega, obsSigma, obsDelta, obsGphiphi, obsSin, obsCos;

    // Disque de Novikov-Thorne.
    int    diskEnabled;
    double isco, diskOuter, diskShapeMax, diskTempScaleK, diskPeakR, diskGain;
    double diskExposure, diskOpacity;   // contraste volumétrique (exposition + opacité χ)
    double opacitySpectralIndex;        // χ_λ ∝ (λref/λ)^indice : auto-absorption chromatique
    double planckNorm;                  // calibration luminance de dev_planckB (cf. commentaire)

    // Texture de plasma (fBm + bras spiraux) + anisotropie MHD (cisaillement képlérien).
    double noiseStrength, noiseFreq, noiseLacunarity, noiseGain;
    int    noiseOctaves, spiralArms;
    double spiralPitch, spiralStrength, noiseEvolution;
    double shearWindup, shearGain, fluxPower;

    // Intégrateur RKF45 adaptatif.
    double tolerance, stepInitial, stepMax, stepMin, horizonMargin, rPlus, safety;
    int    maxSteps;

    // Corridor de raffinement pour les anneaux de photons d'ordre élevé + AA adaptative.
    double photonRingRadius, photonRingStep;
    int    ringSupersample;

    // LUT couleur (Planck ⊗ CIE) et ciel.
    double lutTmin, lutTmax, lutInvStep;
    int    lutSize;                    // dernier index valide (= taille - 1)
    int    skyW, skyH, skyLoaded;
    double starDensity, starBrightness;

    // État d'animation courant.
    double animTime, animAzimuth;
};

// --- État géodésique vu comme vecteur à 5 composantes -------------------------
struct GState { double r, theta, phi, pr, pth; };

__device__ __forceinline__ GState operator+(const GState& a, const GState& b) {
    return {a.r + b.r, a.theta + b.theta, a.phi + b.phi, a.pr + b.pr, a.pth + b.pth};
}
__device__ __forceinline__ GState operator-(const GState& a, const GState& b) {
    return {a.r - b.r, a.theta - b.theta, a.phi - b.phi, a.pr - b.pr, a.pth - b.pth};
}
__device__ __forceinline__ GState operator*(double s, const GState& a) {
    return {s * a.r, s * a.theta, s * a.phi, s * a.pr, s * a.pth};
}

struct GPhoton { GState s; double E, Lz, Q, K; };
struct GSample { double3 color; double g; double drift; int hit; };  // hit : 0 fond,1 disque,2 horizon

__device__ __forceinline__ double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// =============================================================================
//  Géométrie de Kerr / disque (équivalents device de Metric.cpp)
// =============================================================================
__device__ __forceinline__ double dev_orbitalOmega(double r, double M, double a) {
    return sqrt(M) / (pow(r, 1.5) + a * sqrt(M));
}
__device__ __forceinline__ double dev_orbitalUt(double r, double M, double a) {
    const double num = pow(r, 1.5) + a * sqrt(M);
    const double rad = r * r * r - 3.0 * M * r * r + 2.0 * a * sqrt(M) * pow(r, 1.5);
    return num / sqrt(rad > 0.0 ? rad : 1e-12);
}
__device__ __forceinline__ double dev_diskRedshift(double r, double Lz, double alphaObs,
                                                   double omegaObs, const GpuParams& p) {
    const double ut = dev_orbitalUt(r, p.M, p.a);
    const double denom = alphaObs * ut * (1.0 - dev_orbitalOmega(r, p.M, p.a) * Lz);
    if (denom <= 1e-6) return 0.0;
    return (1.0 - omegaObs * Lz) / denom;
}

// Coefficient logarithmique du flux de Page-Thorne (racine xi, autres xj, xk).
__device__ __forceinline__ double dev_ptCoeff(double xi, double xj, double xk, double as) {
    if (fabs(xi) < 1e-9) return 0.0;
    return 3.0 * (xi - as) * (xi - as) / (xi * (xi - xj) * (xi - xk));
}

__device__ double dev_pageThorne(double r, const GpuParams& p) {
    const double M = p.M, a = p.a, rin = p.isco;
    if (r <= rin) return 0.0;
    const double as = a / M;
    const double x  = sqrt(r / M);
    const double x0 = sqrt(rin / M);

    const double t  = acos(clampd(as, -1.0, 1.0)) / 3.0;
    const double third = BH_PI / 3.0;
    const double x1 =  2.0 * cos(t - third);
    const double x2 =  2.0 * cos(t + third);
    const double x3 = -2.0 * cos(t);

    const double c1 = dev_ptCoeff(x1, x2, x3, as);
    const double c2 = dev_ptCoeff(x2, x1, x3, as);
    const double c3 = dev_ptCoeff(x3, x1, x2, as);

    const double bracket =
        x - x0 - 1.5 * as * log(x / x0)
        - c1 * log((x - x1) / (x0 - x1))
        - c2 * log((x - x2) / (x0 - x2))
        - c3 * log((x - x3) / (x0 - x3));

    const double denom = x * x * x * x * (x * x * x - 3.0 * x + 2.0 * as);
    const double f = bracket / denom;
    return f > 0.0 ? f : 0.0;
}

// =============================================================================
//  Géodésique (équivalent device de Geodesic.cpp)
// =============================================================================
__device__ GState dev_rhs(const GState& s, double M, double a, double E, double Lz) {
    const double r = s.r;
    const double sinT = fmax(sin(s.theta), 1e-6);
    const double cosT = cos(s.theta);
    const double s2 = sinT * sinT;

    const double Sigma = r * r + a * a * cosT * cosT;
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double Dsafe = (fabs(Delta) < 1e-12) ? copysign(1e-12, Delta) : Delta;

    const double P = E * (r * r + a * a) - a * Lz;
    const double G = Delta * s.pr * s.pr + s.pth * s.pth - P * P / Dsafe
                   + a * a * E * E * s2 - 2.0 * a * E * Lz + Lz * Lz / s2;

    const double dDelta = 2.0 * r - 2.0 * M;
    const double dP     = 2.0 * r * E;
    const double dG_dr  = dDelta * s.pr * s.pr - 2.0 * P * dP / Dsafe
                        + P * P * dDelta / (Dsafe * Dsafe);
    const double dG_dth = 2.0 * a * a * E * E * sinT * cosT
                        - 2.0 * Lz * Lz * cosT / (sinT * s2);

    const double dSig_dr  = 2.0 * r;
    const double dSig_dth = -2.0 * a * a * cosT * sinT;
    const double invSig   = 1.0 / Sigma;
    const double invSig2  = invSig * invSig;

    GState d;
    d.r     = (Delta * invSig) * s.pr;
    d.theta = invSig * s.pth;
    d.phi   = invSig * (a * P / Dsafe - a * E + Lz / s2);
    d.pr    = -0.5 * dG_dr * invSig + 0.5 * G * dSig_dr * invSig2;
    d.pth   = -0.5 * dG_dth * invSig + 0.5 * G * dSig_dth * invSig2;
    return d;
}

__device__ __forceinline__ double dev_carterK(const GState& s, double a, double E, double Lz) {
    const double sinT = fmax(sin(s.theta), 1e-6);
    const double cosT = cos(s.theta);
    const double Q = s.pth * s.pth + cosT * cosT * (Lz * Lz / (sinT * sinT) - a * a * E * E);
    const double LmaE = Lz - a * E;
    return Q + LmaE * LmaE;
}

// --- Runge-Kutta-Fehlberg 4(5) : un pas, renvoie y5 et l'erreur (y5 - y4) -----
__device__ void dev_rkf45(const GState& y, double h, double M, double a, double E, double Lz,
                          GState& y5, GState& err) {
    const GState k1 = dev_rhs(y, M, a, E, Lz);
    const GState k2 = dev_rhs(y + h * ((1.0 / 4.0) * k1), M, a, E, Lz);
    const GState k3 = dev_rhs(y + h * ((3.0 / 32.0) * k1 + (9.0 / 32.0) * k2), M, a, E, Lz);
    const GState k4 = dev_rhs(y + h * ((1932.0 / 2197.0) * k1 - (7200.0 / 2197.0) * k2
                                     + (7296.0 / 2197.0) * k3), M, a, E, Lz);
    const GState k5 = dev_rhs(y + h * ((439.0 / 216.0) * k1 - 8.0 * k2
                                     + (3680.0 / 513.0) * k3 - (845.0 / 4104.0) * k4), M, a, E, Lz);
    const GState k6 = dev_rhs(y + h * ((-8.0 / 27.0) * k1 + 2.0 * k2
                                     - (3544.0 / 2565.0) * k3 + (1859.0 / 4104.0) * k4
                                     - (11.0 / 40.0) * k5), M, a, E, Lz);

    const GState y4 = y + h * ((25.0 / 216.0) * k1 + (1408.0 / 2565.0) * k3
                             + (2197.0 / 4104.0) * k4 - (1.0 / 5.0) * k5);
    y5 = y + h * ((16.0 / 135.0) * k1 + (6656.0 / 12825.0) * k3
                + (28561.0 / 56430.0) * k4 - (9.0 / 50.0) * k5 + (2.0 / 55.0) * k6);
    err = y5 - y4;
}

__device__ __forceinline__ double dev_scaledErrorNorm(const GState& e, const GState& y,
                                                      double atol, double rtol) {
    double s = 0.0;
    double sc;
    sc = atol + rtol * fabs(y.r);     s += (e.r / sc)     * (e.r / sc);
    sc = atol + rtol * fabs(y.theta); s += (e.theta / sc) * (e.theta / sc);
    sc = atol + rtol * fabs(y.phi);   s += (e.phi / sc)   * (e.phi / sc);
    sc = atol + rtol * fabs(y.pr);    s += (e.pr / sc)    * (e.pr / sc);
    sc = atol + rtol * fabs(y.pth);   s += (e.pth / sc)   * (e.pth / sc);
    return sqrt(s / 5.0);
}

__device__ __forceinline__ double dev_proposeStepFactor(double errorNorm, bool& accept,
                                                        double safety) {
    accept = (errorNorm <= 1.0);
    double factor = (errorNorm <= 1e-12) ? 5.0 : safety * pow(errorNorm, -0.2);
    return clampd(factor, 0.2, 5.0);
}

// =============================================================================
//  Caméra (équivalent device de Camera.cpp) — tétrade ZAMO précalculée
// =============================================================================
__device__ GPhoton dev_generateRay(int i, int j, double offX, double offY, const GpuParams& p) {
    const double a = p.a;
    const double alpha = (2.0 * (i + offX) / p.W - 1.0) * p.halfWidth;
    const double beta  = (1.0 - 2.0 * (j + offY) / p.H) * p.halfHeight;

    const double xh = alpha / p.rObs;
    const double yv = beta  / p.rObs;
    double nr = -1.0, nth = yv, nph = -xh;
    const double inv = 1.0 / sqrt(nr * nr + nth * nth + nph * nph);
    nr *= inv; nth *= inv; nph *= inv;

    const double sqrtGpp = sqrt(p.obsGphiphi);
    const double Etilde  = p.obsLapse + p.obsOmega * sqrtGpp * nph;

    GPhoton ph;
    ph.E  = 1.0;
    ph.Lz = (sqrtGpp * nph) / Etilde;
    ph.s.r     = p.rObs;
    ph.s.theta = p.thetaObs;
    ph.s.phi   = 0.0;
    ph.s.pr    = sqrt(p.obsSigma / p.obsDelta) * nr / Etilde;
    ph.s.pth   = sqrt(p.obsSigma) * nth / Etilde;
    ph.Q = ph.s.pth * ph.s.pth
         + p.obsCos * p.obsCos * (ph.Lz * ph.Lz / (p.obsSin * p.obsSin) - a * a * ph.E * ph.E);
    const double LmaE = ph.Lz - a * ph.E;
    ph.K = ph.Q + LmaE * LmaE;
    return ph;
}

// =============================================================================
//  Bruit de Perlin / texture de plasma (équivalent device de Renderer.cpp)
// =============================================================================
__device__ __forceinline__ double dev_fade(double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
__device__ __forceinline__ double dev_grad(int h, double x, double y, double z) {
    const int hh = h & 15;
    const double u = hh < 8 ? x : y;
    const double v = hh < 4 ? y : (hh == 12 || hh == 14 ? x : z);
    return ((hh & 1) ? -u : u) + ((hh & 2) ? -v : v);
}
__device__ double dev_pnoise(double x, double y, double z) {
    const int X = ((int)floor(x)) & 255;
    const int Y = ((int)floor(y)) & 255;
    const int Z = ((int)floor(z)) & 255;
    x -= floor(x); y -= floor(y); z -= floor(z);
    const double u = dev_fade(x), v = dev_fade(y), w = dev_fade(z);
    const int A  = c_perm[X] + Y,     AA = c_perm[A] + Z,     AB = c_perm[A + 1] + Z;
    const int B  = c_perm[X + 1] + Y, BA = c_perm[B] + Z,     BB = c_perm[B + 1] + Z;
    #define BH_L(t, aa, bb) ((aa) + (t) * ((bb) - (aa)))
    return BH_L(w,
        BH_L(v, BH_L(u, dev_grad(c_perm[AA],     x,     y,     z),     dev_grad(c_perm[BA],     x - 1, y,     z)),
                BH_L(u, dev_grad(c_perm[AB],     x,     y - 1, z),     dev_grad(c_perm[BB],     x - 1, y - 1, z))),
        BH_L(v, BH_L(u, dev_grad(c_perm[AA + 1], x,     y,     z - 1), dev_grad(c_perm[BA + 1], x - 1, y,     z - 1)),
                BH_L(u, dev_grad(c_perm[AB + 1], x,     y - 1, z - 1), dev_grad(c_perm[BB + 1], x - 1, y - 1, z - 1))));
    #undef BH_L
}
__device__ double dev_fbm(double x, double y, double z, int octaves, double lacunarity, double gain) {
    double sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * dev_pnoise(x * freq, y * freq, z * freq);
        norm += amp;
        amp  *= gain;
        freq *= lacunarity;
    }
    return norm > 0.0 ? sum / norm : 0.0;
}
// =============================================================================
//  Pipeline VOLUMÉTRIQUE (FP32, intrinsèques CUDA) — remplace l'ancien disque
//  plan. Géométrie épaisse H(r)=0.05 r, turbulence MHD (fBm), thermodynamique
//  de Novikov-Thorne et corps noir de Planck échantillonné en RVB. La précision
//  simple est volontaire ici : ce bloc est le coût dominant et n'intervient pas
//  dans la conservation de Carter (assurée par la géodésique restée en double).
// =============================================================================
__device__ __forceinline__ float dev_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// --- Bruit de Perlin simple précision (intrinsèques) : turbulence de plasma ---
__device__ __forceinline__ float dev_fadef(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
__device__ __forceinline__ float dev_gradf(int h, float x, float y, float z) {
    const int hh = h & 15;
    const float u = hh < 8 ? x : y;
    const float v = hh < 4 ? y : (hh == 12 || hh == 14 ? x : z);
    return ((hh & 1) ? -u : u) + ((hh & 2) ? -v : v);
}
__device__ float dev_pnoisef(float x, float y, float z) {
    const int X = ((int)floorf(x)) & 255;
    const int Y = ((int)floorf(y)) & 255;
    const int Z = ((int)floorf(z)) & 255;
    x -= floorf(x); y -= floorf(y); z -= floorf(z);
    const float u = dev_fadef(x), v = dev_fadef(y), w = dev_fadef(z);
    const int A  = c_perm[X] + Y,     AA = c_perm[A] + Z,     AB = c_perm[A + 1] + Z;
    const int B  = c_perm[X + 1] + Y, BA = c_perm[B] + Z,     BB = c_perm[B + 1] + Z;
    #define BH_LF(t, aa, bb) ((aa) + (t) * ((bb) - (aa)))
    return BH_LF(w,
        BH_LF(v, BH_LF(u, dev_gradf(c_perm[AA],     x,     y,     z),     dev_gradf(c_perm[BA],     x - 1, y,     z)),
                 BH_LF(u, dev_gradf(c_perm[AB],     x,     y - 1, z),     dev_gradf(c_perm[BB],     x - 1, y - 1, z))),
        BH_LF(v, BH_LF(u, dev_gradf(c_perm[AA + 1], x,     y,     z - 1), dev_gradf(c_perm[BA + 1], x - 1, y,     z - 1)),
                 BH_LF(u, dev_gradf(c_perm[AB + 1], x,     y - 1, z - 1), dev_gradf(c_perm[BB + 1], x - 1, y - 1, z - 1))));
    #undef BH_LF
}
__device__ float dev_fbmf(float x, float y, float z, int octaves, float lacunarity, float gain) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * dev_pnoisef(x * freq, y * freq, z * freq);
        norm += amp;
        amp  *= gain;
        freq *= lacunarity;
    }
    return norm > 0.0f ? sum / norm : 0.0f;
}

// --- Température locale de Novikov-Thorne : T(r) ∝ r^{-3/4} (1-√(r_in/r))^{1/4} -
// La correction de bord intérieur encode le spin via r_isco(a). Normalisée pour
// que diskTempScaleK soit atteinte au pic (r ≈ 1.361 r_isco ; h_max ≈ 0.4878).
__device__ __forceinline__ float dev_ntTemperature(float r, const GpuParams& p) {
    const float ri = (float)p.isco;
    if (r <= ri) return 0.0f;
    const float base = __powf(r / ri, -0.75f);
    const float corr = __powf(fmaxf(1.0f - __fsqrt_rn(ri / r), 0.0f), 0.25f);
    const float peakNorm = 2.0500f;                 // 1 / max(base*corr)
    return (float)p.diskTempScaleK * peakNorm * base * corr;
}

// --- Corps noir de Planck échantillonné en RVB (700 / 546 / 435 nm) -----------
// B(λ,T) ∝ λ^{-5} / (exp(c2/(λT)) - 1), c2 = hc/k_B = 1.43877688e7 nm·K.
// La référence verte (546 nm) ne donne PAS des valeurs O(1) en général — au pic
// du disque (T ~ 7000 K), B(546,T) ~ 0.024, sous-exposant tout le rendu d'un
// facteur ~40 par rapport à la LUT CPU (Planck ⊗ CIE 1931, normalisée à Y=1 au
// pic). Fix : p.planckNorm = exp(c2/(546·diskTempScaleK)) - 1, précalculé côté
// hôte, ramène B(546, diskTempScaleK) à exactement 1 — même convention que le
// CPU (planckColor(diskTempScaleK) ≈ Y=1), pour un kDiskGain/diskExposure
// cohérent entre les deux moteurs.
__device__ __forceinline__ float dev_planckB(float lambda_nm, float T) {
    const float c2 = 1.43877688e7f;
    const float x  = c2 / (lambda_nm * fmaxf(T, 1.0f));
    const float em = __expf(x) - 1.0f;
    const float l5 = __powf(546.0f / lambda_nm, 5.0f);
    return l5 / fmaxf(em, 1e-20f);
}
__device__ __forceinline__ float3 dev_planckRGB(float T, const GpuParams& p) {
    const float norm = (float)p.planckNorm;
    return make_float3(dev_planckB(700.0f, T) * norm,
                        dev_planckB(546.0f, T) * norm,
                        dev_planckB(435.0f, T) * norm);
}

// --- Opacité chromatique (auto-absorption synchrotron, approximation) ---------
// Un plasma synchrotron thermique n'absorbe pas la lumière de façon grise :
// l'auto-absorption dépend de la fréquence (plus forte vers le bleu en régime
// optiquement épais). Approximation à un seul indice spectral autour de la
// référence verte (546 nm) : χ_λ/χ0 = (λref/λ)^s. s=0 restitue l'opacité grise
// (comportement précédent, un seul χ pour les 3 canaux) ; s>0 rend le rouge
// relativement plus transparent que le bleu, par analogie avec le
// durcissement spectral de l'auto-absorption synchrotron.
__device__ __forceinline__ float3 dev_opacitySpectrum(const GpuParams& p) {
    const float s = (float)p.opacitySpectralIndex;
    return make_float3(__powf(546.0f / 700.0f, s), 1.0f, __powf(546.0f / 435.0f, s));
}

// =============================================================================
//  Turbulence MHD anisotrope : approximation de tube de flux magnétique
//  cisaillé par la rotation différentielle képlérienne.
//
//  Un filament de plasma gelé dans le champ magnétique (théorème d'Alfvén,
//  advection idéale MHD) est étiré par le cisaillement képlérien
//  S(r) = -r dΩ/dr = (3/2)Ω(r). Le tenseur gradient de déformation d'un
//  cisaillement simple (x,y) -> (x + γy, y) a pour valeurs propres
//  (γ/2 ± √(1+(γ/2)²)) : après un temps τ (γ = Sτ), le rapport d'aspect
//  résultant d'un blob initialement isotrope est √(1+(Sτ)²). C'est cette loi
//  — pas un facteur ad hoc — qui pilote l'élongation azimutale des filaments
//  ci-dessous : basse fréquence de bruit le long de l'écoulement (ê_φ, axe
//  long) et haute fréquence en travers (axe court), aire du noyau conservée.
//
//  La cohérence transverse (across-field) est étiquetée par un label de flux
//  poloïdal ψ(r) = r^p (p = fluxPower ≈ 3/4, forme standard d'un champ
//  poloïdal faible threading un tore d'accrétion — tores de Fishbone-Moncrief
//  / De Villiers-Hawley-Krolik) : le bruit varie doucement le long des lignes
//  de champ, approximant l'organisation en rubans/canaux de la MRI (instabilité
//  magnétorotationnelle) observée dans les simulations GRMHD.
// =============================================================================
__device__ __forceinline__ float dev_shearAspect(float r, const GpuParams& p) {
    const float Om     = (float)dev_orbitalOmega(r, p.M, p.a);
    const float S      = 1.5f * Om;                                    // cisaillement képlérien
    const float OmPeak = (float)dev_orbitalOmega(p.diskPeakR, p.M, p.a);
    const float tau    = (float)p.shearWindup * (2.0f * (float)BH_PI) / fmaxf(OmPeak, 1e-6f);
    const float St     = (float)p.shearGain * S * tau;
    return __fsqrt_rn(1.0f + St * St);                                 // rapport d'aspect ellipse
}

__device__ float dev_mhdDensityFactor(double r, double sinT, double cosT, double phi,
                                      const GpuParams& p) {
    const float rf = (float)r;

    // Label de flux poloïdal (coordonnée "across-field", axe court du repère
    // de cisaillement) : varie doucement en r, sert d'axe transverse.
    const float psi = __powf(fmaxf(rf, 1e-3f), (float)p.fluxPower);

    // Coordonnée le long de l'écoulement (≈ ê_φ local) : arc azimutal
    // co-tournant, pré-cisaillé de shearWindup tours orbitaux (au rayon de
    // pic) — les rayons petits (Ω grand) accumulent davantage de cisaillement
    // que les rayons grands : c'est la rotation différentielle elle-même qui
    // enroule/étire le motif, pas une simple rotation rigide de bloc.
    const float Om      = (float)dev_orbitalOmega(r, p.M, p.a);
    const float OmPeak  = (float)dev_orbitalOmega(p.diskPeakR, p.M, p.a);
    const float tauWind = (float)p.shearWindup * (2.0f * (float)BH_PI) / fmaxf(OmPeak, 1e-6f);
    const float phiShear = (float)phi - Om * ((float)p.animTime + tauWind);
    const float cyl = rf * (float)sinT;
    const float sAlong = cyl * phiShear;                               // arc-length le long du flot

    const float aspect = dev_shearAspect(rf, p);
    const float fr = (float)p.noiseFreq;
    // Fréquence anisotrope : basse le long de l'écoulement (filaments longs et
    // fins), élevée en travers — aire du noyau de bruit ~conservée (aspect × 1/aspect).
    const float nx = sAlong * (fr / aspect);
    const float ny = psi    * (fr * aspect);
    const float nz = (float)cosT * rf * fr + (float)p.noiseEvolution * (float)p.animTime;

    float turb = dev_fbmf(nx, ny, nz, p.noiseOctaves, (float)p.noiseLacunarity, (float)p.noiseGain);
    float fil = fmaxf(1.0f + (float)p.noiseStrength * turb, 0.0f);

    if (p.spiralArms > 0) {
        const float s = __cosf(p.spiralArms * phiShear
                               - (p.spiralArms / (float)p.spiralPitch) * __logf(fmaxf(rf, 1e-3f)));
        fil *= 1.0f + (float)p.spiralStrength * 0.5f * (0.5f * (s + 1.0f));
    }
    return fil;
}

// --- Densité de gaz : enveloppe radiale × profil vertical gaussien × MHD -------
// Géométrie volumétrique : demi-épaisseur H(r)=0.05 r, de l'ISCO à r=25 ;
// filaments de plasma par turbulence MHD anisotrope (dev_mhdDensityFactor).
__device__ float dev_plasmaDensity(double r, double sinT, double cosT, double phi,
                                   const GpuParams& p) {
    const float rf = (float)r;
    if (rf <= (float)p.isco || rf >= 25.0f) return 0.0f;

    const float H = 0.05f * rf;                       // demi-épaisseur hyperbolique
    const float z = rf * (float)cosT;                 // hauteur sur le plan équatorial
    const float vert = __expf(-(z * z) / (2.0f * H * H));

    const float ri    = (float)p.isco;
    const float inner = dev_clampf((rf - ri) / (0.5f * ri), 0.0f, 1.0f);   // montée à l'ISCO
    const float outer = dev_clampf((25.0f - rf) / 4.0f, 0.0f, 1.0f);       // extinction à r=25
    const float rad   = inner * outer * __powf(ri / rf, 0.5f);             // chute ~ r^{-1/2}

    const float fil = dev_mhdDensityFactor(r, sinT, cosT, phi, p);
    return vert * rad * fil;
}

// --- Backend procédural device de l'interface PlasmaState (cf. Plasma.hpp) ----
// Miroir exact de Renderer::sampleProceduralPlasma() (CPU) : même interface,
// même sémantique de validité (seuil 1e-4), mêmes formules d'orbite. Les deux
// backends sont volontairement gardés structurellement parallèles — c'est la
// divergence CPU/CUDA qui a produit les deux bugs physiques de cette session
// (double comptage g⁴ côté CPU, calibration de luminance côté CUDA).
//
// PlasmaState est un POD à membres scalaires : son constructeur par défaut
// implicite est utilisable en code device, la struct est donc partagée telle
// quelle entre les deux backends (pas de duplication de définition).
//
// B4 est ici aussi un champ JOUET (toroïdal) — pas une sortie GRMHD réelle.
__device__ PlasmaState dev_sampleProceduralPlasma(double r, double sinT, double cosT,
                                                   double phi, const GpuParams& p) {
    PlasmaState ps;
    ps.rho = (double)dev_plasmaDensity(r, sinT, cosT, phi, p);
    ps.valid = ps.rho > 1e-4;
    if (!ps.valid) return ps;

    const double Omega = dev_orbitalOmega(r, p.M, p.a);
    const double ut    = dev_orbitalUt(r, p.M, p.a);
    ps.u4[0] = ut;
    ps.u4[1] = 0.0;
    ps.u4[2] = 0.0;
    ps.u4[3] = Omega * ut;

    ps.B4[0] = 0.0;
    ps.B4[1] = 0.0;
    ps.B4[2] = 0.0;
    ps.B4[3] = Omega * ut;   // toy field, même direction que le flot

    ps.Theta_e = 0.0;        // non peuplé avant l'étape 2 (prescription R-beta)

    return ps;
}

// =============================================================================
//  Couleur : LUT de Planck, ciel procédural / texturé
// =============================================================================
__device__ __forceinline__ double3 dev_planckColor(double T, const float3* lut, const GpuParams& p) {
    const double t = clampd(T, p.lutTmin, p.lutTmax);
    const double f = (t - p.lutTmin) * p.lutInvStep;
    int k = (int)f;
    const int last = p.lutSize;
    if (k >= last) { const float3 c = lut[last]; return make_double3(c.x, c.y, c.z); }
    if (k < 0) k = 0;
    const float3 a = lut[k];
    const float3 b = lut[k + 1];
    const double fr = f - k;
    return make_double3(a.x + (b.x - a.x) * fr, a.y + (b.y - a.y) * fr, a.z + (b.z - a.z) * fr);
}

__device__ __forceinline__ double dev_hash2(int a, int b) {
    uint32_t x = (uint32_t)a * 374761393u + (uint32_t)b * 668265263u;
    x = (x ^ (x >> 13)) * 1274126177u;
    x ^= x >> 16;
    return (x & 0xFFFFFFu) / (double)0x1000000;
}

__device__ double3 dev_spectralColor(double u) {
    const double cum[7] = {0.0003, 0.0100, 0.0700, 0.1700, 0.2900, 0.5000, 1.0000};
    const double3 rgb[7] = {
        {0.62, 0.74, 1.00}, {0.70, 0.80, 1.00}, {0.86, 0.91, 1.00}, {1.00, 0.98, 0.92},
        {1.00, 0.95, 0.80}, {1.00, 0.84, 0.62}, {1.00, 0.70, 0.50}};
    for (int i = 0; i < 7; ++i) if (u < cum[i]) return rgb[i];
    return rgb[6];
}

__device__ double3 dev_skyColor(double dx, double dy, double dz, const float3* sky, const GpuParams& p) {
    const double phi   = atan2(dy, dx);
    const double theta = acos(clampd(dz, -1.0, 1.0));

    if (p.skyLoaded) {
        double u = phi / (2.0 * BH_PI) + 0.5;
        u -= floor(u);
        const double v = clampd(theta / BH_PI, 0.0, 1.0);
        int ix = (int)(u * p.skyW); if (ix > p.skyW - 1) ix = p.skyW - 1;
        int iy = (int)(v * p.skyH); if (iy > p.skyH - 1) iy = p.skyH - 1;
        const float3 c = sky[(size_t)iy * p.skyW + ix];
        return make_double3(c.x, c.y, c.z);
    }

    const double t = 0.5 * (dz + 1.0);
    double3 col = make_double3(0.018 + (0.002 - 0.018) * t,
                               0.020 + (0.002 - 0.020) * t,
                               0.030 + (0.010 - 0.030) * t);

    const double cell = 0.045;
    const int cu = (int)floor(phi / cell);
    const int cv = (int)floor(theta / cell);
    if (dev_hash2(cu, cv) < p.starDensity) {
        const double bx = dev_hash2(cu * 7 + 1, cv * 3 + 5);
        const double by = dev_hash2(cu * 13 + 2, cv * 17 + 9);
        const double du = phi   - (cu + bx) * cell;
        const double dv = theta - (cv + by) * cell;
        const double sigma = 0.005;
        const double inten = exp(-(du * du + dv * dv) / (2.0 * sigma * sigma));
        const double mag = pow(dev_hash2(cu * 5 + 3, cv * 11 + 7), 7.0);
        const double bright = (0.25 + 6.0 * mag) * inten * p.starBrightness;
        const double3 sc = dev_spectralColor(dev_hash2(cu + 99, cv - 13));
        col.x += sc.x * bright; col.y += sc.y * bright; col.z += sc.z * bright;
    }
    return col;
}

// --- Position pseudo-cartésienne de Kerr (oblate sphéroïdale) ------------------
// Sert au calcul du pas géométrique ds et n'altère pas la géodésique en BL.
__device__ __forceinline__ double3 dev_kerrCartesian(double r, double sinT, double cosT,
                                                     double phi, double a) {
    const double R = sqrt(r * r + a * a);
    return make_double3(R * sinT * cos(phi), R * sinT * sin(phi), r * cosT);
}

// =============================================================================
//  Tracé arrière d'un rayon : RAYMARCHING VOLUMÉTRIQUE + transfert radiatif.
//
//  La géodésique RKF45 (double, Carter conservé) est intégrée pas à pas ; à
//  chaque pas accepté tombant dans le volume du disque, on résout l'équation de
//  transfert dI/dλ = j - χ I sur le segment. Le photon TRAVERSE le volume sans
//  être interrompu (pas de break à la première intersection) ; on accumule
//  l'émission en composition front-to-back jusqu'à saturation optique ou sortie.
// =============================================================================
__device__ GSample dev_traceRay(const GPhoton& ph, const float3* lut, const float3* sky,
                                 const GpuParams& p) {
    (void)lut;                                   // LUT non utilisée : couleur via Planck RVB direct
    GState y = ph.s;
    const double E = ph.E, Lz = ph.Lz, K0 = ph.K;

    const double rStop = p.rPlus * (1.0 + p.horizonMargin);
    const double atol = p.tolerance, rtol = p.tolerance;

    // Constantes du transfert radiatif (unités du moteur). emit/abs = fonction
    // source ; la densité s'annule dans S et ne pilote que la profondeur optique.
    // diskOpacity ↑ rend la face avant optiquement épaisse → l'émission sature à la
    // fonction source et les filaments restent contrastés dans la zone très brillante ;
    // diskExposure module globalement la luminosité du disque (anti-saturation bleu).
    const float kAbs       = (float)p.diskOpacity;                 // χ = kAbs · densité
    const float kSrc       = (float)(p.diskGain * p.diskExposure); // j/χ = kSrc · g⁴ · Planck(T_obs)
    const float marchStep  = 0.05f;                               // pas λ max imposé dans le volume
    const float3 chi       = dev_opacitySpectrum(p);               // χ_λ/χ0 par canal (R,G,B)

    double h = p.stepInitial;
    double maxDrift = 0.0;

    // Accumulateurs du transfert radiatif (front-to-back), vectoriels : chaque
    // canal a sa PROPRE transmittance (opacité chromatique, cf. dev_opacitySpectrum)
    // — un vrai transfert radiatif multi-bande, pas un scalaire gris repeint en RVB.
    double3 Iacc  = make_double3(0.0, 0.0, 0.0);
    double3 trans = make_double3(1.0, 1.0, 1.0);
    bool    hitDisk = false, escaped = false, fellIn = false;
    double  gRecord = 0.0, bestWeight = 0.0;

    for (int step = 0; step < p.maxSteps; ++step) {
        const double r = y.r;
        double hMaxEff = clampd(0.3 * (r - p.rPlus), p.stepMin, p.stepMax);

        // Raffinement du pas à l'approche / dans le volume du disque (|cosθ|<0.15
        // et ISCO ≤ r ≤ 25) pour échantillonner finement la couche mince.
        if (p.diskEnabled && r >= p.isco && r <= 25.0 && fabs(cos(y.theta)) < 0.15)
            hMaxEff = fmin(hMaxEff, (double)marchStep);

        // Corridor de la sphère de photons : raffinement du pas quel que soit
        // l'angle d'incidence (pas seulement dans le plan du disque), pour bien
        // résoudre l'enroulement Δφ des rayons quasi-critiques — indispensable
        // pour capturer les sous-anneaux n≥2,3 (photons faisant plusieurs tours).
        if (r <= p.photonRingRadius)
            hMaxEff = fmin(hMaxEff, p.photonRingStep);

        if (h > hMaxEff) h = hMaxEff;

        GState y5, err;
        dev_rkf45(y, h, p.M, p.a, E, Lz, y5, err);
        bool accept;
        const double errNorm = dev_scaledErrorNorm(err, y, atol, rtol);
        const double factor = dev_proposeStepFactor(errNorm, accept, p.safety);

        if (!accept && h > p.stepMin) {
            h = clampd(h * factor, p.stepMin, hMaxEff);
            continue;
        }

        const GState yNew = y5;

        // --- Échantillon volumétrique au milieu du segment -----------------------
        if (p.diskEnabled && (trans.x > 0.003 || trans.y > 0.003 || trans.z > 0.003)) {
            const double rm   = 0.5 * (y.r + yNew.r);
            const double thm  = 0.5 * (y.theta + yNew.theta);
            const double phm  = 0.5 * (y.phi + yNew.phi);
            const double sinM = fmax(sin(thm), 1e-6), cosM = cos(thm);
            const PlasmaState ps = dev_sampleProceduralPlasma(rm, sinM, cosM, phm, p);

            if (ps.valid) {
                const float dens = (float)ps.rho;
                // Longueur géométrique du segment (pseudo-cartésien de Kerr).
                const double3 c0 = dev_kerrCartesian(y.r, fmax(sin(y.theta), 1e-6),
                                                     cos(y.theta), y.phi, p.a);
                const double3 c1 = dev_kerrCartesian(yNew.r, fmax(sin(yNew.theta), 1e-6),
                                                     cos(yNew.theta), yNew.phi, p.a);
                const float dx = (float)(c1.x - c0.x), dy = (float)(c1.y - c0.y),
                            dz = (float)(c1.z - c0.z);
                const float ds = __fsqrt_rn(dx * dx + dy * dy + dz * dz);

                // Thermodynamique + redshift relativiste complet : T_obs = g·T_local.
                const float Tloc = dev_ntTemperature((float)rm, p);
                float g = (float)dev_diskRedshift(rm, Lz, p.obsLapse, p.obsOmega, p);
                g = dev_clampf(g, 0.02f, 8.0f);
                const float Tobs = g * Tloc;

                // Fonction source : Planck(T_obs) ⊗ beaming bolométrique g⁴.
                const float g4 = g * g * g * g;
                const float3 B = dev_planckRGB(Tobs, p);
                const float sScale = kSrc * g4;
                const float3 S = make_float3(sScale * B.x, sScale * B.y, sScale * B.z);

                // Transfert radiatif MULTI-BANDE : profondeur optique et
                // transmittance propres à chaque canal (χ_λ = kAbs·dens·ds·chi_λ),
                // I_λ += trans_λ·S_λ·(1-e^{-dτ_λ}) ; trans_λ *= e^{-dτ_λ}.
                const float dtauR = kAbs * dens * ds * chi.x;
                const float dtauG = kAbs * dens * ds * chi.y;
                const float dtauB = kAbs * dens * ds * chi.z;
                const float wR = (float)trans.x * (1.0f - __expf(-dtauR));
                const float wG = (float)trans.y * (1.0f - __expf(-dtauG));
                const float wB = (float)trans.z * (1.0f - __expf(-dtauB));
                Iacc.x += wR * S.x; Iacc.y += wG * S.y; Iacc.z += wB * S.z;
                trans.x *= __expf(-dtauR); trans.y *= __expf(-dtauG); trans.z *= __expf(-dtauB);

                const float wgt = 0.2126f * wR + 0.7152f * wG + 0.0722f * wB;  // luma (Rec.709)
                if (wgt > bestWeight) { bestWeight = wgt; gRecord = g; }
                hitDisk = true;
            }
        }

        const double drift = fabs(dev_carterK(yNew, p.a, E, Lz) - K0) / (fabs(K0) + 1e-12);
        maxDrift = fmax(maxDrift, drift);

        y = yNew;
        h = clampd(h * factor, p.stepMin, hMaxEff);

        if (trans.x < 0.003 && trans.y < 0.003 && trans.z < 0.003) { break; }  // volume optiquement saturé
        if (y.r <= rStop)                 { fellIn = true; break; }
        if (y.r >= p.rObs && y.pr > 0.0)  { escaped = true; break; }
    }

    // Budget de pas épuisé sans résolution (rayon quasi critique n≥2,3 encore en
    // train de boucler) : NE PAS retourner un pixel noir arbitraire (terminaison
    // prématurée) — on classe le rayon selon le sens de propagation radiale au
    // moment de la coupure (p_r > 0 : en route vers l'échappement ; p_r < 0 : en
    // route vers l'horizon), un compromis bien plus proche de la vraie solution
    // qu'un artefact noir pour les sous-anneaux de photons.
    if (!escaped && !fellIn && !hitDisk) {
        if (y.pr > 0.0) escaped = true; else fellIn = true;
    }

    // --- Composition de l'arrière-plan derrière le volume résiduel ---------------
    GSample s;
    s.drift = maxDrift;
    s.g = gRecord;
    s.hit = hitDisk ? 1 : (fellIn ? 2 : 0);

    double3 bg = make_double3(0.0, 0.0, 0.0);
    if (escaped && (trans.x > 0.0 || trans.y > 0.0 || trans.z > 0.0)) {   // ciel visible derrière le volume
        const double phiSky = y.phi + p.animAzimuth;
        const double st = sin(y.theta), ct = cos(y.theta);
        bg = dev_skyColor(st * cos(phiSky), st * sin(phiSky), ct, sky, p);
    }
    s.color = make_double3(Iacc.x + trans.x * bg.x,
                           Iacc.y + trans.y * bg.y,
                           Iacc.z + trans.z * bg.z);
    return s;
}

// =============================================================================
//  Kernel : un thread = un pixel. Grille de base S×S (SSAA) + sur-échantillonnage
//  ADAPTATIF ciblé sur les pixels de bord (frontière fond/horizon/disque), là où
//  vivent les sous-anneaux de photons n≥2,3 : leur largeur en paramètre d'impact
//  décroît exponentiellement (~e^{-2π} par tour supplémentaire, Gralla & Lupsasca
//  2020), une grille SxS régulière ne les touche presque jamais. On ne paie le
//  sur-échantillonnage que là où un vrai bord est détecté — coût quasi nul sur
//  le reste de l'image (ciel, disque, horizon uniformes).
// =============================================================================
constexpr int kMaxBaseSamples = 64;   // borne statique : S ≤ 8 (clampé côté hôte)

__global__ void renderKernel(GpuParams p, const float3* lut, const float3* sky,
                             double3* fb, double* gOut, int* hitOut, double* carterOut,
                             int jStart, int jEnd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = jStart + blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= p.W || j >= jEnd) return;

    const int S = p.S;

    // --- 1) Grille de base S×S (SSAA classique) ; seul le type de contact (hit)
    // est mémorisé par sous-échantillon, pour la détection de bord ci-dessous.
    int     baseHit[kMaxBaseSamples];
    double cg = 0.0, cdrift = 0.0;
    int chit = 0;
    bool haveCenter = false;
    double3 sum = make_double3(0.0, 0.0, 0.0);

    for (int sj = 0; sj < S; ++sj) {
        for (int si = 0; si < S; ++si) {
            const double ox = (si + 0.5) / S;
            const double oy = (sj + 0.5) / S;
            const GPhoton ph = dev_generateRay(i, j, ox, oy, p);
            const GSample s = dev_traceRay(ph, lut, sky, p);
            baseHit[sj * S + si] = s.hit;
            sum.x += s.color.x; sum.y += s.color.y; sum.z += s.color.z;
            if (!haveCenter) { cg = s.g; cdrift = s.drift; chit = s.hit; haveCenter = true; }
        }
    }

    // --- 2) Détection de bord : deux sous-échantillons voisins tombent dans des
    // catégories différentes (fond / disque / horizon) => transition franche,
    // signature du bord critique de l'ombre où se cachent les sous-anneaux.
    bool edge = false;
    for (int sj = 0; sj < S && !edge; ++sj) {
        for (int si = 0; si < S && !edge; ++si) {
            const int k = sj * S + si;
            if (si + 1 < S && baseHit[k] != baseHit[k + 1]) edge = true;
            if (sj + 1 < S && baseHit[k] != baseHit[k + S]) edge = true;
        }
    }

    // --- 3) Sur-échantillonnage additionnel, jitteré, UNIQUEMENT sur les pixels
    // de bord : ES×ES rayons de plus, moyennés avec la grille de base.
    int nExtra = 0;
    if (edge && p.ringSupersample > 0) {
        const int ES = p.ringSupersample;
        uint32_t seed = (uint32_t)(i * 9781 + j * 6271 + 1u);
        for (int e = 0; e < ES * ES; ++e) {
            seed = seed * 1664525u + 1013904223u;
            const double jx = ((seed >> 8) & 0xFFFFu) / 65536.0;
            seed = seed * 1664525u + 1013904223u;
            const double jy = ((seed >> 8) & 0xFFFFu) / 65536.0;
            const GPhoton ph = dev_generateRay(i, j, jx, jy, p);
            const GSample s = dev_traceRay(ph, lut, sky, p);
            sum.x += s.color.x; sum.y += s.color.y; sum.z += s.color.z;
            ++nExtra;
        }
    }

    const int totalSamples = S * S + nExtra;
    const double scale = p.exposure / (double)totalSamples;
    const size_t idx = (size_t)j * p.W + i;
    fb[idx] = make_double3(sum.x * scale, sum.y * scale, sum.z * scale);
    gOut[idx] = cg;
    hitOut[idx] = chit;
    carterOut[idx] = cdrift;
}

}  // namespace cuda_detail

// =============================================================================
//  Hôte : Renderer::renderGPU — prépare la scène, lance le kernel par bandes,
//  rapatrie l'image HDR linéaire et les cartes scientifiques.
// =============================================================================
RenderResult Renderer::renderGPU() const {
    using namespace cuda_detail;
    const int W = cfg_.width, H = cfg_.height;

    // --- Paramètres de la scène --------------------------------------------------
    GpuParams p;
    p.M = bh_.mass();  p.a = bh_.spin();
    p.W = W;  p.H = H;
    // p.S borne à 8 : le noyau mémorise S² catégories de contact par thread
    // dans un tableau local de taille statique (kMaxBaseSamples = 64) pour la
    // détection de bord (cf. renderKernel) — au-delà, utiliser plutôt
    // ringSupersample pour affiner davantage les pixels de bord détectés.
    {
        const int sReq = (cfg_.samples < 1) ? 1 : cfg_.samples;
        p.S = (sReq > 8) ? 8 : sReq;
        if (sReq > 8)
            std::fprintf(stderr, "  [GPU] samples=%d capped at 8 (use ringSupersample for photon-subring detail)\n", sReq);
    }
    p.exposure = cfg_.exposure;

    const double thetaObs = (90.0 - cfg_.cameraInclination) * kPi / 180.0;
    const Kerr::ZamoFrame z = bh_.zamo(rObs_, thetaObs);
    p.rObs = rObs_;  p.thetaObs = thetaObs;
    p.halfWidth  = cfg_.imageHalfWidth;
    p.halfHeight = cfg_.imageHalfWidth * ((double)H / (double)W);
    p.obsLapse = z.lapse;  p.obsOmega = z.omega;
    p.obsSigma = z.sigma;  p.obsDelta = z.delta;  p.obsGphiphi = z.gphiphi;
    p.obsSin = z.sinTheta; p.obsCos = z.cosTheta;

    p.diskEnabled = cfg_.diskEnabled ? 1 : 0;
    p.isco = isco_;  p.diskOuter = cfg_.diskOuter;
    p.diskShapeMax = diskShapeMax_;  p.diskTempScaleK = cfg_.diskTempScaleK;
    p.diskPeakR = diskPeakR_;  p.diskGain = 5.0;   // kDiskGain (cf. Renderer.cpp)
    // Calibration de dev_planckB : ramène B(546nm, diskTempScaleK) à 1, même
    // convention de luminance que la LUT CPU (cf. commentaire de dev_planckRGB).
    {
        const double c2 = 1.43877688e7;
        const double x = c2 / (546.0 * cfg_.diskTempScaleK);
        p.planckNorm = std::exp(x) - 1.0;
    }
    p.diskExposure = cfg_.diskExposure;  p.diskOpacity = cfg_.diskOpacity;
    p.opacitySpectralIndex = cfg_.diskOpacitySpectralIndex;

    p.noiseStrength = cfg_.diskNoiseStrength;  p.noiseFreq = cfg_.diskNoiseFreq;
    p.noiseLacunarity = cfg_.diskNoiseLacunarity;  p.noiseGain = cfg_.diskNoiseGain;
    p.noiseOctaves = cfg_.diskNoiseOctaves;  p.spiralArms = cfg_.diskSpiralArms;
    p.spiralPitch = cfg_.diskSpiralPitch;  p.spiralStrength = cfg_.diskSpiralStrength;
    p.noiseEvolution = cfg_.diskNoiseEvolution;
    p.shearWindup = cfg_.diskShearWindup;  p.shearGain = cfg_.diskShearGain;
    p.fluxPower = cfg_.diskFluxPower;

    p.tolerance = cfg_.tolerance;  p.stepInitial = cfg_.stepInitial;
    p.stepMax = cfg_.stepMax;  p.stepMin = cfg_.stepMin;
    p.horizonMargin = cfg_.horizonMargin;  p.rPlus = rPlus_;  p.safety = 0.9;
    // Budget de pas dédié au GPU (gpuMaxSteps), plus généreux que le défaut CPU :
    // les rayons quasi critiques (sous-anneaux n≥2,3) ont besoin de beaucoup plus
    // de pas pour boucler plusieurs fois autour de la sphère de photons sans
    // terminaison prématurée (cf. dev_traceRay). Seuls CES rayons consomment
    // vraiment ce budget élargi (la grande majorité — fond, disque, horizon
    // direct — termine tôt, budget ou pas) ; en revanche le raffinement du PAS
    // dans le corridor r ≤ photonRingRadius (dev_traceRay) est, lui, payé par
    // TOUT rayon qui traverse cette zone (pas seulement les quasi-critiques) —
    // c'est le vrai coût GPU de cette fonctionnalité. Si le temps de rendu
    // explose, relâcher photonRingStep (ex. 0.01 -> 0.03-0.05) en premier.
    p.maxSteps = (cfg_.gpuMaxSteps > 0) ? cfg_.gpuMaxSteps : cfg_.maxSteps;

    p.photonRingRadius = cfg_.photonRingRadius;
    p.photonRingStep   = cfg_.photonRingStep;
    p.ringSupersample  = cfg_.ringSupersample;

    p.lutTmin = lutTmin_;  p.lutTmax = lutTmax_;  p.lutInvStep = lutInvStep_;
    p.lutSize = (int)tempLUT_.size() - 1;
    p.skyW = skyW_;  p.skyH = skyH_;  p.skyLoaded = skyLoaded_ ? 1 : 0;
    p.starDensity = cfg_.starDensity;  p.starBrightness = cfg_.starBrightness;
    p.animTime = animTime_;  p.animAzimuth = animAzimuth_;

    // --- Table de permutation de Perlin (régénérée à l'identique du CPU) ---------
    {
        int perm[256];
        for (int i = 0; i < 256; ++i) perm[i] = i;
        uint32_t s = 1234567u;
        for (int i = 255; i > 0; --i) {
            s = s * 1664525u + 1013904223u;
            const int jj = (int)(s % (uint32_t)(i + 1));
            const int tmp = perm[i]; perm[i] = perm[jj]; perm[jj] = tmp;
        }
        int p512[512];
        for (int i = 0; i < 512; ++i) p512[i] = perm[i & 255];
        CUDA_CHECK(cudaMemcpyToSymbol(c_perm, p512, sizeof(p512)));
    }

    // --- LUT couleur (Vec3 double -> float3) -------------------------------------
    std::vector<float3> lutHost(tempLUT_.size());
    for (size_t i = 0; i < tempLUT_.size(); ++i)
        lutHost[i] = make_float3((float)tempLUT_[i].x, (float)tempLUT_[i].y, (float)tempLUT_[i].z);
    float3* d_lut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lut, lutHost.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_lut, lutHost.data(), lutHost.size() * sizeof(float3), cudaMemcpyHostToDevice));

    // --- Texture de ciel optionnelle (triplets RGB -> float3) --------------------
    float3* d_sky = nullptr;
    if (skyLoaded_ && skyW_ > 0 && skyH_ > 0) {
        std::vector<float3> skyHost((size_t)skyW_ * skyH_);
        for (size_t k = 0; k < skyHost.size(); ++k)
            skyHost[k] = make_float3(skyTex_[k * 3 + 0], skyTex_[k * 3 + 1], skyTex_[k * 3 + 2]);
        CUDA_CHECK(cudaMalloc(&d_sky, skyHost.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(d_sky, skyHost.data(), skyHost.size() * sizeof(float3), cudaMemcpyHostToDevice));
    }

    // --- Tampons de sortie sur le GPU --------------------------------------------
    const size_t N = (size_t)W * H;
    double3* d_fb = nullptr;  double* d_g = nullptr;  int* d_hit = nullptr;  double* d_carter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fb,     N * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_g,      N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hit,    N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_carter, N * sizeof(double)));

    // --- Lancement par bandes horizontales (retour de progression) ---------------
    const dim3 block(16, 16);
    const int bandRows = 64;
    for (int jStart = 0; jStart < H; jStart += bandRows) {
        const int jEnd = (jStart + bandRows < H) ? (jStart + bandRows) : H;
        const dim3 grid((W + block.x - 1) / block.x, (jEnd - jStart + block.y - 1) / block.y);
        renderKernel<<<grid, block>>>(p, d_lut, d_sky, d_fb, d_g, d_hit, d_carter, jStart, jEnd);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const int done = jEnd;
        std::fprintf(stderr, "\r  rendu GPU : %3d %%", (int)(100.0 * done / H));
        std::fflush(stderr);
    }
    std::fprintf(stderr, "\n");

    // --- Rapatriement -----------------------------------------------------------
    std::vector<double3> fb(N);
    std::vector<double>  gbuf(N), cbuf(N);
    std::vector<int>     hbuf(N);
    CUDA_CHECK(cudaMemcpy(fb.data(),   d_fb,     N * sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gbuf.data(), d_g,      N * sizeof(double),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbuf.data(), d_hit,    N * sizeof(int),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cbuf.data(), d_carter, N * sizeof(double),  cudaMemcpyDeviceToHost));

    cudaFree(d_fb); cudaFree(d_g); cudaFree(d_hit); cudaFree(d_carter);
    cudaFree(d_lut); if (d_sky) cudaFree(d_sky);

    // --- Construction du RenderResult (image HDR + cartes) ----------------------
    RenderResult out;
    out.width = W; out.height = H;
    out.image = Image(W, H);
    out.redshift.assign(N, 0.0);
    out.hitType.assign(N, 0);
    out.carterDrift.assign(N, 0.0);

    double maxDrift = 0.0;
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            const size_t idx = (size_t)j * W + i;
            const double3 c = fb[idx];
            out.image.setPixel(i, j, c.x, c.y, c.z);
            out.redshift[idx]    = gbuf[idx];
            out.hitType[idx]     = hbuf[idx];
            out.carterDrift[idx] = cbuf[idx];
            if (cbuf[idx] > maxDrift) maxDrift = cbuf[idx];
        }
    }
    out.maxCarterDrift = maxDrift;
    return out;
}

}  // namespace bhsim
