#pragma once
#include <string>

namespace bhsim {

// Constante pi portable (M_PI n'est pas garanti par la norme).
inline constexpr double kPi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Paramètres de la simulation.
// Unités géométrisées G = c = 1 ; la masse fixe l'échelle (M = 1 par défaut).
// Le spin est donné en unités de M : spin = a/M ∈ [0, 1) (a = moment cinétique).
// -----------------------------------------------------------------------------
struct Config {
    // --- Image ---
    int width   = 1920;
    int height  = 1080;         // 16:9 Full HD
    int samples = 4;            // anti-aliasing : S x S rayons par pixel (4 -> 16 spp)
    double exposure = 0.6;      // exposition globale avant tone-mapping

    // --- Trou noir (Kerr) ---
    double M    = 1.0;          // masse
    double spin = 0.9;          // a/M  (0 = Schwarzschild, ->1 = extrémal)

    // --- Caméra (observateur lointain, plan image en paramètres d'impact) ---
    double cameraDistance    = 1000.0;  // r_obs (grand : relations de Bardeen exactes)
    double cameraInclination = 12.0;    // angle (deg) au-dessus du plan équatorial
    double imageHalfWidth    = 26.0;    // demi-largeur du plan image (en M)

    // --- Disque d'accrétion de Novikov-Thorne (plan équatorial, du ISCO au bord) ---
    bool   diskEnabled    = true;
    double diskOuter      = 22.0;       // bord externe (en M) ; bord interne = ISCO
    double diskTempScaleK = 7000.0;     // température (K) au pic du profil
    double dopplerExponent = 4.0;       // amplification relativiste (g^4 bolométrique)

    // --- Exposition / opacité volumétrique (contraste du plasma) ----------------
    // Le flanc gauche (blueshifté, beaming g^4) sature en blanc pur si le volume
    // reste optiquement mince. Augmenter diskOpacity rend la face avant optiquement
    // épaisse : l'émission sature à la fonction source et les filaments de
    // turbulence redeviennent discernables dans la zone ultra-lumineuse.
    double diskExposure = 0.8;          // multiplicateur d'exposition du disque (GPU)
    double diskOpacity  = 1.5;          // χ = diskOpacity · densité (1 = ancien rendu mince)

    // --- Texture de plasma du disque (bruit de Perlin 3D relativiste) -----------
    // Le disque homogène est modulé par une turbulence fractale (fBm) couplée au
    // rayon r et à l'angle φ, co-tournée à la vitesse képlérienne relativiste
    // Ω(r) = √M/(r^{3/2}+a√M) en fonction du temps t : filaments, bras spiraux.
    double diskNoiseStrength = 0.85;    // amplitude de modulation (0 = disque lisse)
    double diskNoiseFreq     = 0.55;    // fréquence spatiale de base du bruit (1/M)
    int    diskNoiseOctaves  = 5;       // octaves de fBm (détail des filaments)
    double diskNoiseLacunarity = 2.15;  // facteur de fréquence entre octaves
    double diskNoiseGain     = 0.55;    // décroissance d'amplitude entre octaves
    int    diskSpiralArms    = 2;       // nombre de bras spiraux (0 = aucun)
    double diskSpiralPitch   = 2.6;     // serrage logarithmique des bras
    double diskSpiralStrength = 0.45;   // contraste des bras spiraux
    // Évolution temporelle du champ de bruit (en plus de la co-rotation rigide) :
    // décale la phase fractale au fil des frames pour une turbulence « vivante »
    // (filaments qui se forment et se dissipent), pas seulement tournante.
    double diskNoiseEvolution = 0.015;  // vitesse d'évolution du bruit (par unité de temps M)

    // --- Ciel étoilé d'arrière-plan --------------------------------------------
    // Si starTexturePath pointe vers un BMP 24 bits équirectangulaire, il est
    // échantillonné comme sphère céleste (lentillée gravitationnellement). Sinon,
    // un générateur procédural en loi de puissance (types spectraux O..M) le remplace.
    std::string starTexturePath = "";   // "" = étoiles procédurales (loi de puissance)
    double starDensity   = 0.10;        // probabilité d'étoile par cellule du ciel
    double starBrightness = 1.0;        // gain global du champ d'étoiles

    // --- Animation (séquence vidéo image par image) ----------------------------
    double time         = 0.0;          // phase temporelle (rotation du disque), en M
    double cameraAzimuth = 0.0;         // azimut de la caméra sur son orbite ZAMO (rad)
    bool   animate      = false;        // mode animation (généré via la CLI)
    int    animFrames   = 120;          // nombre d'images de la séquence
    double animCameraOrbitDeg = 24.0;   // amplitude du déplacement azimutal sur la séquence
    double animDiskTurns = 1.0;         // tours képlériens du disque (au rayon de pic) sur la séquence

    // --- Intégration géodésique (RKF45 adaptatif, hamiltonien canonique) ---
    double tolerance    = 1e-10;  // tolérances atol = rtol du contrôleur de pas (Carter < 1e-6 partout)
    double stepInitial  = 1.0;
    double stepMax      = 50.0;   // plafond du pas (le pas est aussi borné à ~0.3(r-r+))
    double stepMin      = 1e-6;
    int    maxSteps     = 200000;
    double horizonMargin = 1e-3;  // arrêt à r+ * (1 + marge)  (évite Δ -> 0)

    // --- Vérification de la constante de Carter ---
    double carterTolerance = 1e-6;  // dérive max admissible de K le long du rayon

    // --- Sortie ---
    std::string outputDir = "output";
    bool writeImage    = true;
    bool writeMatrices = true;
};

}  // namespace bhsim
