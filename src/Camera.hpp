#pragma once
#include "Geodesic.hpp"
#include "Metric.hpp"
#include "Config.hpp"

namespace bhsim {

struct Photon {
    State     state;
    Constants constants;
};

class Camera {
public:
    Camera(const Config& cfg, const Kerr& bh);

    Photon generateRay(int i, int j, double offX = 0.5, double offY = 0.5) const;

    double observerRadius() const { return rObs_; }
    double observerLapse()  const { return obsLapse_; }
    double observerOmega()  const { return obsOmega_; }

private:
    Kerr   bh_;
    double rObs_, thetaObs_;
    int    width_, height_;
    double halfWidth_, halfHeight_;
    double obsLapse_ = 1.0, obsOmega_ = 0.0;
};

}  // namespace bhsim
