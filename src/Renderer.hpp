#pragma once
#include <vector>
#include "Config.hpp"
#include "Metric.hpp"
#include "Camera.hpp"
#include "Image.hpp"
#include "Plasma.hpp"
#include "Vec3.hpp"

namespace bhsim {

enum class Hit { Background = 0, Disk = 1, Horizon = 2 };

// Résultat du rendu : image + matrices scientifiques.
struct RenderResult {
    Image image;
    std::vector<double> redshift;     // facteur g sur le disque ; 0 sinon
    std::vector<int>    hitType;      // voir enum Hit
    std::vector<double> carterDrift;  // dérive relative max de K le long du rayon
    double maxCarterDrift = 0.0;      // dérive maximale sur toute l'image
    int width = 0, height = 0;
};

// -----------------------------------------------------------------------------
// Moteur de rendu Kerr par tracé de géodésiques arrière (hamiltonien canonique).
// -----------------------------------------------------------------------------
class Renderer {
public:
    explicit Renderer(const Config& cfg);
    RenderResult render() const;

#ifdef BHSIM_CUDA
    // Rendu GPU (CUDA) : intégrateur RKF45 de Kerr + disque de Novikov-Thorne
    // parallélisés (un thread par pixel). Défini dans src/cuda/Renderer.cu.
    // Produit un RenderResult identique à render() (mêmes formules, double précision).
    RenderResult renderGPU() const;
#endif

    // État d'animation : phase temporelle t (rotation képlérienne du disque) et
    // azimut de la caméra sur son orbite ZAMO. À régler avant chaque render().
    void setFrame(double time, double cameraAzimuth) {
        animTime_ = time;
        animAzimuth_ = cameraAzimuth;
    }

private:
    struct Sample {
        Vec3   color;
        double g = 0.0;
        double carterDrift = 0.0;
        Hit    hit = Hit::Background;
    };

    Sample traceRay(const Photon& photon) const;

    // Densité de plasma (volumétrique, sans dimension) au point (r, θ, φ) :
    // enveloppe radiale (ISCO -> bord) × profil vertical gaussien (demi-épaisseur
    // H(r) = 0.05 r) × turbulence/bras spiraux. Nulle hors du tore du disque.
    double diskDensity(double r, double sinTheta, double cosTheta, double phi) const;

    // Backend procédural de l'interface PlasmaState (cf. Plasma.hpp) : peuple
    // rho (= diskDensity), la quadrivitesse d'orbite képlérienne circulaire
    // (formules exactes déjà utilisées ailleurs dans le moteur — pas de
    // nouvelle physique) et un champ B jouet. Point d'entrée unique que la
    // boucle de tracé interroge — un futur backend GRMHD implémentera la même
    // interface sans toucher au reste du pipeline.
    PlasmaState sampleProceduralPlasma(double r, double sinTheta, double cosTheta, double phi) const;

    // Fonction source du transfert radiatif en (r, Lz) : corps noir de Page-Thorne
    // à la température perçue g·T_em, amplifié par le beaming bolométrique g⁴.
    // Renvoie aussi g (facteur de décalage spectral) via gOut, pour redshift.csv.
    Vec3 diskSource(double r, double Lz, double& gOut) const;

    // Modulation de plasma (turbulence fBm + bras spiraux) du disque au point
    // (r, φ), co-tournée à la vitesse képlérienne relativiste à l'instant t.
    double diskTexture(double r, double phi) const;

    // Ciel étoilé pour la direction d'évasion d̂ : texture équirectangulaire si
    // chargée, sinon champ procédural en loi de puissance (types spectraux).
    Vec3   skyColor(const Vec3& dir) const;

    // Couleur sRGB linéaire (HDR : couleur + luminance physiques) d'un corps noir
    // à la température T, par interpolation dans la table Planck ⊗ CMF CIE 1931.
    Vec3   planckColor(double T) const;

    Config cfg_;
    Kerr   bh_;
    Camera camera_;

    // Grandeurs précalculées.
    double isco_ = 0.0;
    double rPlus_ = 0.0;
    double rObs_ = 0.0;
    double diskShapeMax_ = 1.0;   // normalisation du flux de Page-Thorne
    double obsLapse_ = 1.0;       // tétrade ZAMO de la caméra : lapse α
    double obsOmega_ = 0.0;       // tétrade ZAMO de la caméra : entraînement ω

    // Table température (K) -> couleur sRGB linéaire physique (loi de Planck
    // intégrée contre les fonctions colorimétriques CIE 1931), normalisée à
    // luminance Y = 1 pour la température de pic du disque.
    std::vector<Vec3> tempLUT_;
    double lutTmin_ = 800.0, lutTmax_ = 40000.0, lutInvStep_ = 0.0;

    // État d'animation courant (rotation du disque + orbite caméra).
    double animTime_ = 0.0;
    double animAzimuth_ = 0.0;

    // Rayon de référence de la texture (pic du flux) : normalise la rotation.
    double diskPeakR_ = 6.0;

    // Texture équirectangulaire d'arrière-plan (BMP 24 bits), si chargée.
    std::vector<float> skyTex_;       // RGB linéaire, taille 3*skyW_*skyH_
    int skyW_ = 0, skyH_ = 0;
    bool skyLoaded_ = false;
};

}  // namespace bhsim
