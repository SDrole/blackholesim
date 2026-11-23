#pragma once
#include <string>

namespace bhsim {

// Portable pi constant (M_PI is not guaranteed by the C++ standard).
inline constexpr double kPi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Simulation parameters.
// Geometrized units G = c = 1; the mass sets the scale (M = 1 by default).
// Spin is specified in units of M: spin = a/M in [0,1).
// -----------------------------------------------------------------------------
struct Config {
    // --- Image ---
    int width   = 1920;
    int height  = 1080;         // 16:9 Full HD
    int samples = 4;            // anti-aliasing: S x S rays per pixel (4 -> 16 spp)
    double exposure = 0.6;      // global exposure before tone mapping

    // --- Black hole (Kerr) ---
    double M    = 1.0;          // mass
    double spin = 0.9;          // a/M (0 = Schwarzschild, ->1 = extremal)

    // --- Camera (distant observer, image plane in impact parameters) ---
    double cameraDistance    = 1000.0;  // r_obs; large enough for Bardeen relations
    double cameraInclination = 12.0;    // degrees above the equatorial plane
    double imageHalfWidth    = 26.0;    // image-plane half width, in M

    // --- Novikov-Thorne accretion disk ---
    bool   diskEnabled     = true;
    double diskOuter       = 22.0;      // outer edge in M; inner edge = ISCO
    double diskTempScaleK  = 7000.0;    // temperature at profile peak, kelvin
    double dopplerExponent = 4.0;       // bolometric relativistic boosting exponent

    // --- Volumetric exposure / opacity ---
    // In an optically thin volume the strongly blueshifted side can saturate.
    // Increasing diskOpacity makes the front layer optically thicker so
    // emission approaches the source function and turbulent structure remains
    // visible in very bright regions.
    double diskExposure = 0.8;           // disk exposure multiplier (GPU)
    double diskOpacity  = 1.2;           // chi = diskOpacity * density
    double diskScaleHeightRatio = 0.025; // H(r)/r geometric torus half-thickness

    // --- Chromatic opacity / simplified synchrotron self-absorption ---
    // Thermal synchrotron absorption is frequency dependent. This simplified
    // approximation uses chi_lambda = chi0 * (lambda_ref/lambda)^index with
    // lambda_ref = 546 nm. Setting the index to zero recovers gray opacity.
    double diskOpacitySpectralIndex = 2.0;

    // --- Anisotropic MHD-inspired turbulence (GPU) ---
    // Frozen-in plasma structures are stretched by differential Keplerian
    // shear S(r)=(3/2)Omega(r). A simple shear deformation gives an aspect
    // ratio sqrt(1+(S tau)^2), which is used here instead of an arbitrary
    // elongation factor. Cross-field coherence uses a poloidal flux label
    // psi(r) proportional to r^diskFluxPower.
    double diskShearWindup = 4.0;       // pre-existing shear age in peak-orbit turns
    double diskShearGain   = 1.0;       // shear -> anisotropy coupling strength
    double diskFluxPower   = 0.75;      // poloidal flux-label exponent

    // --- Procedural plasma texture ---
    // The smooth disk is modulated by 3D fractal noise coupled to r and phi and
    // co-rotated at the relativistic Keplerian angular velocity
    // Omega(r)=sqrt(M)/(r^(3/2)+a sqrt(M)).
    double diskNoiseStrength = 0.85;    // 0 = smooth disk
    double diskNoiseFreq     = 0.55;    // base spatial frequency, 1/M
    int    diskNoiseOctaves  = 5;       // fBm octaves
    double diskNoiseLacunarity = 2.15;  // frequency ratio between octaves
    double diskNoiseGain     = 0.55;    // amplitude falloff between octaves
    int    diskSpiralArms    = 2;       // 0 = no spiral modulation
    double diskSpiralPitch   = 2.6;     // logarithmic winding
    double diskSpiralStrength = 0.45;   // spiral-arm contrast
    // Time evolution of the noise field in addition to rigid co-rotation.
    double diskNoiseEvolution = 0.015;  // evolution rate per M of coordinate time

    // --- Instrument response / point-spread function ---
    // Real imaging systems have a finite PSF. The HDR linear image is
    // convolved before tone mapping with a Gaussian kernel and mixed with the
    // original using a bounded weight, so the pass redistributes light instead
    // of inventing arbitrary bloom energy.
    bool   glowEnabled    = true;
    double glowRadiusFrac = 0.006;  // Gaussian sigma as a fraction of image width
    double glowStrength   = 0.12;   // 0 = disabled, 1 = fully blurred contribution

    // --- Background star field ---
    // If starTexturePath points to a 24-bit equirectangular BMP it is sampled
    // as the lensed celestial sphere. Otherwise a procedural power-law stellar
    // population is used.
    std::string starTexturePath = "";   // empty = procedural stars
    double starDensity    = 0.10;       // star probability per sky cell
    double starBrightness = 1.0;        // global stellar-field gain

    // --- Animation ---
    double time          = 0.0;         // disk-evolution phase, in M
    double cameraAzimuth = 0.0;         // camera azimuth around the hole, rad
    bool   animate       = false;       // CLI animation mode
    int    animFrames    = 120;         // number of output frames
    double animCameraOrbitDeg = 24.0;   // total camera azimuth excursion
    double animDiskTurns = 1.0;         // peak-radius Keplerian turns over sequence

    // --- Geodesic integration (adaptive RKF45, canonical Hamiltonian) ---
    double tolerance    = 1e-10;  // atol=rtol target; Carter drift <1e-6 in validation
    double stepInitial  = 1.0;
    double stepMax      = 50.0;   // also limited near the horizon
    double stepMin      = 1e-6;
    int    maxSteps     = 200000;
    double horizonMargin = 1e-3;  // stop at r+*(1+margin) to avoid Delta -> 0

    // --- Higher-order photon rings (GPU) ---
    // Near-critical impact parameters can orbit the photon region multiple
    // times before escaping or being captured. They need a larger step budget,
    // finer integration throughout the photon-region corridor, and targeted
    // supersampling because successive subrings become exponentially narrow.
    int    gpuMaxSteps      = 1500000;
    double photonRingRadius = 6.0;     // refined corridor radius in M
    double photonRingStep   = 0.005;   // maximum step inside that corridor
    int    ringSupersample  = 8;       // additional supersampling on edge pixels

    // --- Carter constant validation ---
    double carterTolerance = 1e-6;     // maximum accepted relative drift of K

    // --- Output ---
    std::string outputDir = "output";
    bool writeImage    = true;
    bool writeMatrices = true;
};

}  // namespace bhsim
