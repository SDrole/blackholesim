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

struct RenderResult {
    Image image;
    std::vector<double> redshift;
    std::vector<int>    hitType;      // voir enum Hit
    std::vector<double> carterDrift;
    double maxCarterDrift = 0.0;
    int width = 0, height = 0;
};

class Renderer {
public:
    explicit Renderer(const Config& cfg);
    RenderResult render() const;

#ifdef BHSIM_CUDA
    RenderResult renderGPU() const;
#endif

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

    double diskDensity(double r, double sinTheta, double cosTheta, double phi) const;

    PlasmaState sampleProceduralPlasma(double r, double sinTheta, double cosTheta, double phi) const;

    Vec3 diskSource(double r, double Lz, double& gOut) const;

    double diskTexture(double r, double phi) const;

    Vec3   skyColor(const Vec3& dir) const;

    Vec3   planckColor(double T) const;

    Config cfg_;
    Kerr   bh_;
    Camera camera_;

    double isco_ = 0.0;
    double rPlus_ = 0.0;
    double rObs_ = 0.0;
    double diskShapeMax_ = 1.0;   // normalisation du flux de Page-Thorne
    double obsLapse_ = 1.0;
    double obsOmega_ = 0.0;

    std::vector<Vec3> tempLUT_;
    double lutTmin_ = 800.0, lutTmax_ = 40000.0, lutInvStep_ = 0.0;

    double animTime_ = 0.0;
    double animAzimuth_ = 0.0;

    double diskPeakR_ = 6.0;

    std::vector<float> skyTex_;
    int skyW_ = 0, skyH_ = 0;
    bool skyLoaded_ = false;
};

}  // namespace bhsim
