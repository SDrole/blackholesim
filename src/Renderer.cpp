#include "Renderer.hpp"
#include "Geodesic.hpp"
#include "Integrator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>

namespace bhsim {

namespace {

constexpr double kDiskGain = 5.0;   // gain de luminosité du disque (HDR, avant tone-mapping)

// --- Rayonnement de corps noir : loi de Planck intégrée contre les CMF CIE 1931 ---
//
// Approximation analytique multi-lobes des fonctions colorimétriques CIE 1931
// (Wyman, Sloan & Shirley, JCGT 2013) : g(λ;μ,σ1,σ2) = exp(-½t²) avec
// t = (λ-μ)/σ, σ = σ1 si λ<μ sinon σ2 ; λ en nanomètres.
double gaussLobe(double l, double mu, double s1, double s2) {
    const double t = (l - mu) / (l < mu ? s1 : s2);
    return std::exp(-0.5 * t * t);
}
double cieX(double l) {
    return 1.056 * gaussLobe(l, 599.8, 37.9, 31.0)
         + 0.362 * gaussLobe(l, 442.0, 16.0, 26.7)
         - 0.065 * gaussLobe(l, 501.1, 20.4, 26.2);
}
double cieY(double l) {
    return 0.821 * gaussLobe(l, 568.8, 46.9, 40.5)
         + 0.286 * gaussLobe(l, 530.9, 16.3, 31.1);
}
double cieZ(double l) {
    return 1.217 * gaussLobe(l, 437.0, 11.8, 36.0)
         + 0.681 * gaussLobe(l, 459.0, 26.0, 13.8);
}

// Radiance spectrale de Planck (échelle arbitraire) : B_λ ∝ λ⁻⁵ / (e^{c2/λT} - 1).
double planckRad(double lam_nm, double T) {
    constexpr double c2 = 1.43877e7;            // hc/k_B en nm·K
    const double x = c2 / (lam_nm * T);
    if (x > 700.0) return 0.0;                  // évite l'overflow de exp
    const double l2 = lam_nm * lam_nm;
    return 1.0 / (l2 * l2 * lam_nm * std::expm1(x));
}

// Tristimulus XYZ (CIE 1931) du corps noir à T : intégration sur le visible.
Vec3 planckXYZ(double T) {
    double X = 0.0, Y = 0.0, Z = 0.0;
    constexpr double lo = 360.0, hi = 830.0, d = 5.0;
    for (double l = lo; l <= hi; l += d) {
        const double b = planckRad(l, T);
        X += b * cieX(l); Y += b * cieY(l); Z += b * cieZ(l);
    }
    return Vec3(X, Y, Z) * d;
}

// XYZ (illuminant D65) -> sRGB linéaire.
Vec3 xyzToLinSRGB(const Vec3& c) {
    return Vec3( 3.2406 * c.x - 1.5372 * c.y - 0.4986 * c.z,
                -0.9689 * c.x + 1.8758 * c.y + 0.0415 * c.z,
                 0.0557 * c.x - 0.2040 * c.y + 1.0570 * c.z);
}

double hash2(int a, int b) {
    uint32_t x = static_cast<uint32_t>(a) * 374761393u
               + static_cast<uint32_t>(b) * 668265263u;
    x = (x ^ (x >> 13)) * 1274126177u;
    x ^= x >> 16;
    return (x & 0xFFFFFFu) / static_cast<double>(0x1000000);
}

// --- Bruit de Perlin gradient 3D (Ken Perlin, "improved noise" 2002) ----------
// Utilisé pour texturer le disque : turbulence de plasma, filaments. La table de
// permutation est figée (déterministe) ; le bruit est C¹, valeurs dans ~[-1,1].
struct Perlin3D {
    std::array<int, 512> p;
    Perlin3D() {
        std::array<int, 256> perm;
        for (int i = 0; i < 256; ++i) perm[i] = i;
        // Mélange déterministe (LCG) — reproductible d'une exécution à l'autre.
        uint32_t s = 1234567u;
        for (int i = 255; i > 0; --i) {
            s = s * 1664525u + 1013904223u;
            const int j = static_cast<int>(s % static_cast<uint32_t>(i + 1));
            std::swap(perm[i], perm[j]);
        }
        for (int i = 0; i < 512; ++i) p[i] = perm[i & 255];
    }
    static double fade(double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
    static double grad(int h, double x, double y, double z) {
        const int hh = h & 15;
        const double u = hh < 8 ? x : y;
        const double v = hh < 4 ? y : (hh == 12 || hh == 14 ? x : z);
        return ((hh & 1) ? -u : u) + ((hh & 2) ? -v : v);
    }
    double noise(double x, double y, double z) const {
        const int X = static_cast<int>(std::floor(x)) & 255;
        const int Y = static_cast<int>(std::floor(y)) & 255;
        const int Z = static_cast<int>(std::floor(z)) & 255;
        x -= std::floor(x); y -= std::floor(y); z -= std::floor(z);
        const double u = fade(x), v = fade(y), w = fade(z);
        const int A  = p[X] + Y,   AA = p[A] + Z,  AB = p[A + 1] + Z;
        const int B  = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
        auto L = [](double t, double a, double b) { return a + t * (b - a); };
        return L(w,
            L(v, L(u, grad(p[AA],   x,   y,   z),   grad(p[BA],   x-1, y,   z)),
                 L(u, grad(p[AB],   x,   y-1, z),   grad(p[BB],   x-1, y-1, z))),
            L(v, L(u, grad(p[AA+1], x,   y,   z-1), grad(p[BA+1], x-1, y,   z-1)),
                 L(u, grad(p[AB+1], x,   y-1, z-1), grad(p[BB+1], x-1, y-1, z-1))));
    }
};

const Perlin3D& perlin() { static const Perlin3D inst; return inst; }

// Bruit fractal (fBm) : somme d'octaves de Perlin, fréquence ×lacunarity et
// amplitude ×gain à chaque octave. Renvoie une valeur ~[-1,1].
double fbm(double x, double y, double z, int octaves, double lacunarity, double gain) {
    double sum = 0.0, amp = 1.0, norm = 0.0, freq = 1.0;
    const Perlin3D& pn = perlin();
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * pn.noise(x * freq, y * freq, z * freq);
        norm += amp;
        amp  *= gain;
        freq *= lacunarity;
    }
    return norm > 0.0 ? sum / norm : 0.0;
}

// Couleur (sRGB linéaire) d'une étoile d'un type spectral tiré aléatoirement
// selon la fonction de masse initiale (loi de puissance : rares chaudes/bleues
// O,B,A ; nombreuses froides/rouges K,M). u ∈ [0,1) sélectionne le type.
Vec3 spectralColor(double u) {
    // Fréquences relatives approchées (séquence principale du voisinage solaire).
    // O/B/A/F/G/K/M cumulés — dominés par K et M.
    struct SC { double cum; Vec3 rgb; };
    static const SC table[] = {
        { 0.0003, Vec3(0.62, 0.74, 1.00) },  // O — bleu
        { 0.0100, Vec3(0.70, 0.80, 1.00) },  // B — bleu-blanc
        { 0.0700, Vec3(0.86, 0.91, 1.00) },  // A — blanc
        { 0.1700, Vec3(1.00, 0.98, 0.92) },  // F — blanc-jaune
        { 0.2900, Vec3(1.00, 0.95, 0.80) },  // G — jaune (Soleil)
        { 0.5000, Vec3(1.00, 0.84, 0.62) },  // K — orange
        { 1.0000, Vec3(1.00, 0.70, 0.50) },  // M — rouge
    };
    for (const SC& s : table) if (u < s.cum) return s.rgb;
    return table[6].rgb;
}

}  // namespace

Renderer::Renderer(const Config& cfg)
    : cfg_(cfg), bh_(cfg.M, cfg.M * cfg.spin), camera_(cfg, bh_) {
    isco_  = bh_.iscoPrograde();
    rPlus_ = bh_.horizonOuter();
    rObs_  = cfg.cameraDistance;
    obsLapse_ = camera_.observerLapse();   // tétrade ZAMO de la caméra (distance finie)
    obsOmega_ = camera_.observerOmega();
    animTime_    = cfg_.time;              // état d'animation initial (image fixe)
    animAzimuth_ = cfg_.cameraAzimuth;

    // Normalisation du profil de flux de Page-Thorne (pic sur [ISCO, bord]) et
    // rayon de pic (référence pour la vitesse angulaire de rotation de la texture).
    diskShapeMax_ = 1e-30;
    diskPeakR_ = isco_ * 1.5;
    const int N = 4000;
    for (int k = 0; k <= N; ++k) {
        const double r = isco_ * 1.0001 + (cfg_.diskOuter - isco_ * 1.0001) * k / N;
        const double f = bh_.pageThorneFlux(r);
        if (f > diskShapeMax_) { diskShapeMax_ = f; diskPeakR_ = r; }
    }

    // Table température -> couleur sRGB linéaire physique (loi de Planck intégrée
    // contre les CMF CIE 1931). On normalise par la luminance Y du corps noir à
    // la température de pic du disque : un pixel au pic (g = 1) a donc Y = 1, les
    // zones plus chaudes (beaming Doppler) dépassent 1 -> HDR géré par l'ACES.
    const int NL = 2048;
    lutInvStep_ = NL / (lutTmax_ - lutTmin_);
    const Vec3 xyzRef = planckXYZ(cfg_.diskTempScaleK);
    const double invYref = (xyzRef.y > 1e-30) ? (1.0 / xyzRef.y) : 1.0;
    tempLUT_.resize(NL + 1);
    for (int k = 0; k <= NL; ++k) {
        const double T = lutTmin_ + (lutTmax_ - lutTmin_) * k / NL;
        Vec3 rgb = xyzToLinSRGB(planckXYZ(T) * invYref);
        rgb.x = std::max(0.0, rgb.x);            // clamp hors-gamut (couleurs saturées)
        rgb.y = std::max(0.0, rgb.y);
        rgb.z = std::max(0.0, rgb.z);
        tempLUT_[k] = rgb;
    }

    // Chargement optionnel d'une carte stellaire équirectangulaire (BMP 24 bits).
    // Décodée en sRGB linéaire (inverse gamma 2.2) pour rester cohérente avec le
    // pipeline HDR ; à défaut, on retombe sur le champ d'étoiles procédural.
    if (!cfg_.starTexturePath.empty()) {
        std::ifstream f(cfg_.starTexturePath, std::ios::binary);
        if (f) {
            uint8_t hdr[54];
            f.read(reinterpret_cast<char*>(hdr), 54);
            if (f && hdr[0] == 'B' && hdr[1] == 'M') {
                auto u32 = [&](int o) {
                    return static_cast<uint32_t>(hdr[o]) | (static_cast<uint32_t>(hdr[o+1]) << 8)
                         | (static_cast<uint32_t>(hdr[o+2]) << 16) | (static_cast<uint32_t>(hdr[o+3]) << 24);
                };
                const uint32_t off = u32(10);
                const int w = static_cast<int>(u32(18));
                const int h = static_cast<int>(u32(22));
                const uint16_t bpp = static_cast<uint16_t>(hdr[28] | (hdr[29] << 8));
                if (w > 0 && h > 0 && bpp == 24) {
                    f.seekg(off, std::ios::beg);
                    const int rowSize = (w * 3 + 3) & ~3;
                    std::vector<uint8_t> row(rowSize);
                    skyTex_.assign(static_cast<size_t>(w) * h * 3, 0.0f);
                    skyW_ = w; skyH_ = h;
                    for (int j = 0; j < h; ++j) {
                        f.read(reinterpret_cast<char*>(row.data()), rowSize);
                        if (!f) { skyTex_.clear(); skyW_ = skyH_ = 0; break; }
                        const int jj = h - 1 - j;             // BMP : bas -> haut
                        for (int i = 0; i < w; ++i) {
                            const double b = row[i*3+0] / 255.0;
                            const double g = row[i*3+1] / 255.0;
                            const double r = row[i*3+2] / 255.0;
                            const size_t idx = (static_cast<size_t>(jj) * w + i) * 3;
                            skyTex_[idx+0] = static_cast<float>(std::pow(r, 2.2));
                            skyTex_[idx+1] = static_cast<float>(std::pow(g, 2.2));
                            skyTex_[idx+2] = static_cast<float>(std::pow(b, 2.2));
                        }
                    }
                    skyLoaded_ = !skyTex_.empty();
                }
            }
        }
        if (!skyLoaded_)
            std::fprintf(stderr, "  [sky] could not read texture '%s' -> using procedural stars\n",
                         cfg_.starTexturePath.c_str());
    }
}

Vec3 Renderer::planckColor(double T) const {
    const double t = std::clamp(T, lutTmin_, lutTmax_);
    const double f = (t - lutTmin_) * lutInvStep_;
    int k = static_cast<int>(f);
    const int last = static_cast<int>(tempLUT_.size()) - 1;
    if (k >= last) return tempLUT_[last];
    if (k < 0)     k = 0;
    return lerp(tempLUT_[k], tempLUT_[k + 1], f - k);
}

// Modulation de plasma au point équatorial (r, φ). Le motif co-tourne à la
// vitesse angulaire képlérienne relativiste Ω(r) = √M/(r^{3/2}+a√M) : on l'évalue
// dans le repère co-mobile φ_co = φ - Ω(r)·t. Combine une turbulence fBm 3D
// (filaments) et des bras spiraux logarithmiques. Renvoie un facteur >= 0
// d'espérance ~1 (1 = disque homogène d'origine).
double Renderer::diskTexture(double r, double phi) const {
    const double Omega = bh_.orbitalOmega(r);
    const double phiCo = phi - Omega * animTime_;

    // Coordonnées cartésiennes co-mobiles (continuité automatique en φ). Le
    // cisaillement différentiel (Ω dépend de r) étire naturellement les filaments
    // en arcs : structure spirale émergente du disque de Novikov-Thorne.
    const double fr = cfg_.diskNoiseFreq;
    const double nx = r * std::cos(phiCo) * fr;
    const double ny = r * std::sin(phiCo) * fr;
    // Strates radiales + dérive temporelle (filaments qui se forment/dissipent,
    // au-delà de la simple co-rotation rigide).
    const double nz = std::log(std::max(r, 1e-3)) * 1.3 + cfg_.diskNoiseEvolution * animTime_;

    double turb = fbm(nx, ny, nz, cfg_.diskNoiseOctaves,
                      cfg_.diskNoiseLacunarity, cfg_.diskNoiseGain);   // ~[-1,1]
    turb = std::pow(std::fabs(turb), 0.7) * (turb < 0.0 ? -1.0 : 1.0); // contraste des filaments

    // Bras spiraux logarithmiques : r = r0 e^{(φ_co)/pitch}  ->  motif en
    // cos(m φ_co - (m/pitch) ln r).
    double arms = 0.0;
    if (cfg_.diskSpiralArms > 0) {
        const double s = std::cos(cfg_.diskSpiralArms * phiCo
                                  - (cfg_.diskSpiralArms / cfg_.diskSpiralPitch) * std::log(std::max(r, 1e-3)));
        arms = cfg_.diskSpiralStrength * (0.5 * (s + 1.0) - 0.5);   // ∈ [-strength/2, +strength/2]
    }

    const double f = 1.0 + cfg_.diskNoiseStrength * turb + arms;
    return std::max(0.05, f);   // jamais totalement noir (plancher de plasma ténu)
}

// Densité volumétrique du disque : tore géométriquement épais (et non plus un
// plan infiniment mince). Enveloppe radiale (montée à l'ISCO, coupure au bord)
// × profil vertical gaussien de demi-épaisseur H(r) = 0.05 r (H/r = 5%, ordre
// de grandeur standard d'un disque mince relativiste) × turbulence/bras
// spiraux (diskTexture, réutilisée telle quelle comme facteur de densité —
// c'est la densité de plasma qui varie, pas la température).
double Renderer::diskDensity(double r, double sinTheta, double cosTheta, double phi) const {
    (void)sinTheta;
    if (r <= isco_ || r >= cfg_.diskOuter) return 0.0;

    const double H = cfg_.diskScaleHeightRatio * r;
    const double z = r * cosTheta;
    const double vert = std::exp(-(z * z) / (2.0 * H * H));

    const double inner = std::clamp((r - isco_) / (0.5 * isco_), 0.0, 1.0);
    const double outer = std::clamp((cfg_.diskOuter - r) / 4.0, 0.0, 1.0);
    const double rad = inner * outer * std::sqrt(isco_ / r);

    return vert * rad * diskTexture(r, phi);
}

// Backend procédural de l'interface PlasmaState (cf. Plasma.hpp). Pure
// plomberie : rho reprend diskDensity() telle quelle (même seuil de validité
// 1e-4 que l'ancien test inline dans traceRay), zéro nouvelle physique.
//
// La quadrivitesse u^mu est celle d'une orbite circulaire képlérienne
// équatoriale exacte (Ω(r) et u^t déjà calculés ailleurs dans Metric.cpp,
// utilisés ici tels quels — pas une approximation nouvelle). Le champ B est
// en revanche un JOUET (toroïdal, aligné sur l'écoulement, magnitude
// arbitraire) : aucun backend GRMHD réel ne l'alimente encore, donc B4 n'est
// pas utilisé par le pipeline de rendu actuel (réservé aux étapes 3-4 :
// coefficients synchrotron / polarisation, cf. docs/SESSION_LOG_AND_ROADMAP.md).
PlasmaState Renderer::sampleProceduralPlasma(double r, double sinTheta, double cosTheta, double phi) const {
    PlasmaState ps;
    ps.rho = diskDensity(r, sinTheta, cosTheta, phi);
    ps.valid = ps.rho > 1e-4;
    if (!ps.valid) return ps;

    const double Omega = bh_.orbitalOmega(r);
    const double ut = bh_.orbitalUt(r);
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

// Fonction source du transfert radiatif : corps noir de Page-Thorne à la
// température perçue T_obs = g·T_em. Le beaming bolométrique (Stefan-Boltzmann
// L ∝ T_obs⁴, invariance relativiste de I_ν/ν³) est DÉJÀ porté par la LUT
// planckColor (intégrale de Planck, qui croît en ~T⁴) : ne PAS multiplier par
// un g⁴ supplémentaire ici (bug corrigé — la version initiale, copiée du
// facteur explicite du kernel CUDA qui utilise un Planck RVB non calibré,
// comptait le beaming deux fois et saturait tout en blanc).
Vec3 Renderer::diskSource(double r, double Lz, double& gOut) const {
    const double rel = std::clamp(bh_.pageThorneFlux(r) / diskShapeMax_, 0.0, 1.0);
    const double Tem = cfg_.diskTempScaleK * std::pow(rel, 0.25);

    double g = bh_.diskRedshift(r, Lz, obsLapse_, obsOmega_);
    g = std::clamp(g, 0.02, 8.0);                       // garde-fou (faisceau tangent)
    gOut = g;

    return planckColor(g * Tem) * (kDiskGain * cfg_.diskExposure);
}

// Position pseudo-cartésienne de Kerr (sphéroïdale oblate) : sert uniquement à
// mesurer la longueur géométrique ds d'un segment de rayon pour le transfert
// radiatif (profondeur optique dτ = χ ds) ; n'altère pas la géodésique en BL.
static Vec3 kerrCartesian(double r, double sinTheta, double cosTheta, double phi, double a) {
    const double R = std::sqrt(r * r + a * a);
    return Vec3(R * sinTheta * std::cos(phi), R * sinTheta * std::sin(phi), r * cosTheta);
}

// Couleur du ciel pour la direction d'évasion d̂ (déjà tournée de l'azimut
// caméra). Échantillonne la texture équirectangulaire si chargée, sinon génère
// un champ d'étoiles procédural en loi de puissance (types spectraux O..M). Dans
// les deux cas la distorsion gravitationnelle (anneau d'Einstein) est préservée
// car d̂ est la direction réellement intégrée le long de la géodésique.
Vec3 Renderer::skyColor(const Vec3& d) const {
    const double phi   = std::atan2(d.y, d.x);
    const double theta = std::acos(std::clamp(d.z, -1.0, 1.0));

    if (skyLoaded_) {
        double u = phi / (2.0 * kPi) + 0.5;            // [0,1)
        u -= std::floor(u);
        const double v = std::clamp(theta / kPi, 0.0, 1.0);
        const int ix = std::min(skyW_ - 1, static_cast<int>(u * skyW_));
        const int iy = std::min(skyH_ - 1, static_cast<int>(v * skyH_));
        const size_t idx = (static_cast<size_t>(iy) * skyW_ + ix) * 3;
        return Vec3(skyTex_[idx+0], skyTex_[idx+1], skyTex_[idx+2]);
    }

    // Fond galactique ténu (gradient bleu sombre selon la latitude).
    const double t = 0.5 * (d.z + 1.0);
    Vec3 col = lerp(Vec3(0.018, 0.020, 0.030), Vec3(0.002, 0.002, 0.010), t);

    // Cellules angulaires : une étoile éventuelle par cellule, magnitude tirée en
    // loi de puissance (m = u^p : beaucoup de ternes, rares très brillantes).
    const double cell = 0.045;
    const int cu = static_cast<int>(std::floor(phi / cell));
    const int cv = static_cast<int>(std::floor(theta / cell));

    if (hash2(cu, cv) < cfg_.starDensity) {
        const double bx = hash2(cu * 7 + 1, cv * 3 + 5);
        const double by = hash2(cu * 13 + 2, cv * 17 + 9);
        const double du = phi   - (cu + bx) * cell;
        const double dv = theta - (cv + by) * cell;
        const double sigma = 0.005;
        const double inten = std::exp(-(du * du + dv * dv) / (2.0 * sigma * sigma));
        // Loi de puissance de luminosité (exposant élevé -> queue de brillantes).
        const double mag = std::pow(hash2(cu * 5 + 3, cv * 11 + 7), 7.0);
        const double bright = (0.25 + 6.0 * mag) * inten * cfg_.starBrightness;
        const Vec3 sc = spectralColor(hash2(cu + 99, cv - 13));
        col += sc * bright;
    }
    return col;
}

// Tracé arrière d'un rayon : intégration géodésique RKF45 (inchangée) + à
// chaque pas accepté qui traverse le tore du disque, transfert radiatif
// dI/dλ = j - χI résolu sur le segment (composition front-to-back). Le photon
// TRAVERSE le volume sans s'arrêter au premier contact : on accumule
// l'émission jusqu'à saturation optique (trans -> 0) ou sortie de la scène.
// Remplace l'ancien modèle de disque infiniment mince (un seul croisement du
// plan équatorial) par un tore géométriquement épais, cohérent avec l'aspect
// « gaz lumineux » attendu plutôt qu'un anneau peint à plat.
Renderer::Sample Renderer::traceRay(const Photon& photon) const {
    State y = photon.state;
    const Constants c = photon.constants;
    const double K0 = c.K;
    const double a  = bh_.spin();

    auto f = [this, &c](const State& s) { return geodesicRHS(s, bh_, c); };

    StepControl ctl;
    ctl.hMin = cfg_.stepMin;
    ctl.hMax = cfg_.stepMax;

    const double rStop = rPlus_ * (1.0 + cfg_.horizonMargin);
    const double atol = cfg_.tolerance, rtol = cfg_.tolerance;
    const double kAbs = cfg_.diskOpacity;               // χ = kAbs · densité
    const double marchStep = 0.05;                      // pas λ max imposé dans le volume

    double h = cfg_.stepInitial;
    double maxDrift = 0.0;
    bool hitDisk = false, fellIn = false, escaped = false;

    // Accumulateurs du transfert radiatif (front-to-back).
    Vec3 Iacc(0.0, 0.0, 0.0);
    double trans = 1.0;                                 // transmittance restante exp(-τ)
    double gRecord = 0.0, bestWeight = 0.0;

    for (int step = 0; step < cfg_.maxSteps; ++step) {
        const double r = y.r;
        // Pas borné par la distance à l'horizon : fin près de r+, large au loin.
        double hMaxEff = std::clamp(0.3 * (r - rPlus_), cfg_.stepMin, cfg_.stepMax);
        // Raffinement du pas dans la couche mince du disque (|cosθ| petit) pour
        // échantillonner correctement le profil vertical gaussien.
        if (cfg_.diskEnabled && r >= isco_ && r <= cfg_.diskOuter && std::fabs(std::cos(y.theta)) < 0.15)
            hMaxEff = std::min(hMaxEff, marchStep);
        if (h > hMaxEff) h = hMaxEff;

        const RKF45Result res = rkf45Step(y, h, f);
        bool accept;
        const double errNorm = scaledErrorNorm(res.err, y, atol, rtol);
        const double factor = proposeStepFactor(errNorm, accept, ctl);

        if (!accept && h > cfg_.stepMin) {
            h = std::clamp(h * factor, cfg_.stepMin, hMaxEff);
            continue;                                  // pas rejeté : on raffine
        }

        const State yNew = res.y;

        // --- Échantillon volumétrique au milieu du segment ---
        if (cfg_.diskEnabled && trans > 0.003) {
            const double rm  = 0.5 * (y.r + yNew.r);
            const double thm = 0.5 * (y.theta + yNew.theta);
            const double phm = 0.5 * (y.phi + yNew.phi);
            const double sinM = std::max(std::sin(thm), 1e-6), cosM = std::cos(thm);
            const PlasmaState ps = sampleProceduralPlasma(rm, sinM, cosM, phm);

            if (ps.valid) {
                const double dens = ps.rho;
                // Longueur géométrique du segment (pseudo-cartésien de Kerr).
                const Vec3 p0 = kerrCartesian(y.r,    std::max(std::sin(y.theta), 1e-6),    std::cos(y.theta),    y.phi,    a);
                const Vec3 p1 = kerrCartesian(yNew.r, std::max(std::sin(yNew.theta), 1e-6), std::cos(yNew.theta), yNew.phi, a);
                const double ds = (p1 - p0).norm();

                double g = 0.0;
                const Vec3 S = diskSource(rm, c.Lz, g);   // fonction source j/χ

                // Transfert radiatif : I += trans·S·(1-e^{-dτ}) ; trans·= e^{-dτ}.
                const double dtau = kAbs * dens * ds;
                const double wgt  = trans * (1.0 - std::exp(-dtau));
                Iacc += S * wgt;
                trans *= std::exp(-dtau);

                if (wgt > bestWeight) { bestWeight = wgt; gRecord = g; }
                hitDisk = true;
            }
        }

        // Vérification de la constante de Carter sur le nouvel état.
        const double drift = std::fabs(carterK(yNew, bh_, c) - K0) / (std::fabs(K0) + 1e-12);
        maxDrift = std::max(maxDrift, drift);

        y = yNew;
        h = std::clamp(h * factor, cfg_.stepMin, hMaxEff);

        if (trans < 0.003)              { break; }                 // volume optiquement saturé
        if (y.r <= rStop)               { fellIn  = true; break; }
        if (y.r >= rObs_ && y.pr > 0.0) { escaped = true; break; }
    }

    Sample s;
    s.carterDrift = maxDrift;
    s.g = gRecord;
    s.hit = hitDisk ? Hit::Disk : (fellIn ? Hit::Horizon : Hit::Background);

    Vec3 bg(0.0, 0.0, 0.0);
    if (escaped && trans > 0.0) {
        // Azimut caméra : orbite ZAMO simulée par une rotation rigide du ciel
        // autour de l'axe de spin (axisymétrie de Kerr).
        const double phiSky = y.phi + animAzimuth_;
        const double st = std::sin(y.theta), ct = std::cos(y.theta);
        const Vec3 dir(st * std::cos(phiSky), st * std::sin(phiSky), ct);
        bg = skyColor(dir);
    }
    s.color = Iacc + bg * trans;   // ciel résiduel derrière le volume, atténué
    return s;
}

RenderResult Renderer::render() const {
    const int W = cfg_.width, H = cfg_.height;
    RenderResult out;
    out.width = W; out.height = H;
    out.image = Image(W, H);
    out.redshift.assign(static_cast<size_t>(W) * H, 0.0);
    out.hitType.assign(static_cast<size_t>(W) * H, 0);
    out.carterDrift.assign(static_cast<size_t>(W) * H, 0.0);

    const int S = std::max(1, cfg_.samples);
    const double invSS = 1.0 / static_cast<double>(S * S);
    int rowsDone = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            Vec3 color(0.0, 0.0, 0.0);
            Sample center;
            bool haveCenter = false;
            for (int sj = 0; sj < S; ++sj) {
                for (int si = 0; si < S; ++si) {
                    const double ox = (si + 0.5) / S;
                    const double oy = (sj + 0.5) / S;
                    const Sample s = traceRay(camera_.generateRay(i, j, ox, oy));
                    color += s.color;
                    if (!haveCenter) { center = s; haveCenter = true; }
                }
            }
            color = color * (invSS * cfg_.exposure);
            out.image.setPixel(i, j, color.x, color.y, color.z);
            const size_t idx = static_cast<size_t>(j) * W + i;
            out.redshift[idx]    = center.g;
            out.hitType[idx]     = static_cast<int>(center.hit);
            out.carterDrift[idx] = center.carterDrift;
        }

        int done;
#ifdef _OPENMP
#pragma omp critical
#endif
        { ++rowsDone; done = rowsDone; }

        const int stepRows = std::max(1, H / 20);
        if (done % stepRows == 0 || done == H) {
            std::fprintf(stderr, "\r  rendu : %3d %%", static_cast<int>(100.0 * done / H));
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "\n");

    // Réponse instrumentale (PSF) : appliquée sur le HDR linéaire, avant tout
    // tone-mapping (cf. Config::glow* — convolution à énergie conservée).
    if (cfg_.glowEnabled)
        out.image.applyPointSpreadFunction(cfg_.glowRadiusFrac, cfg_.glowStrength);

    for (double d : out.carterDrift) out.maxCarterDrift = std::max(out.maxCarterDrift, d);
    return out;
}

}  // namespace bhsim
