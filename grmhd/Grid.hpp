#pragma once
#include <vector>
#include <cstddef>
#include "Coordinates.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// S0 — MKS grid for the GRMHD solver.
//
// Finite-volume convention: state variables live at CELL CENTERS; fluxes live
// on FACES. Both coordinate sets are represented explicitly.
//
// Deliberate standard choice: the inner radial boundary lies INSIDE the event
// horizon. Causality prevents physical information from propagating back out,
// so the inner boundary condition cannot contaminate the exterior solution.
// Horizon-penetrating Kerr-Schild coordinates make this possible.
//
// Ghost-cell halo: NG=3 covers PLM and leaves room for future PPM/WENO work
// without changing the storage layout.
// -----------------------------------------------------------------------------
struct Grid {
    // Number of physical cells, excluding ghosts.
    int n1 = 0, n2 = 0, n3 = 0;
    static constexpr int NG = 3;          // ghost cells on each side

    // Domain bounds in MKS coordinates.
    double x1min = 0.0, x1max = 0.0;      // ln r
    double x2min = 0.0, x2max = 1.0;      // parameterized theta
    double x3min = 0.0, x3max = 2.0 * kPi;

    double dx1 = 0.0, dx2 = 0.0, dx3 = 0.0;

    ModifiedKerrSchild coords{0.3};

    // Allocated dimensions, including ghost cells.
    int t1() const { return n1 + 2 * NG; }
    int t2() const { return n2 + 2 * NG; }
    int t3() const { return n3 + 2 * NG; }
    std::size_t cellCount() const {
        return static_cast<std::size_t>(t1()) * t2() * t3();
    }

    // Linear index for i,j,k including ghost cells, 0-based.
    std::size_t idx(int i, int j, int k) const {
        return (static_cast<std::size_t>(i) * t2() + j) * t3() + k;
    }

    // MKS coordinates at cell centers.
    double x1c(int i) const { return x1min + (i - NG + 0.5) * dx1; }
    double x2c(int j) const { return x2min + (j - NG + 0.5) * dx2; }
    double x3c(int k) const { return x3min + (k - NG + 0.5) * dx3; }

    // MKS coordinates of the lower face of a cell.
    double x1f(int i) const { return x1min + (i - NG) * dx1; }
    double x2f(int j) const { return x2min + (j - NG) * dx2; }
    double x3f(int k) const { return x3min + (k - NG) * dx3; }

    // Corresponding physical Kerr-Schild coordinates.
    double rOf(int i) const { return coords.rOf(x1c(i)); }
    double thetaOf(int j) const { return coords.thetaOf(x2c(j)); }
    double phiOf(int k) const { return x3c(k); }

    // Construct a spherical MKS grid. rIn should lie inside the horizon.
    static Grid makeSpherical(int n1, int n2, int n3,
                              double rIn, double rOut, double hslope = 0.3) {
        Grid g;
        g.n1 = n1; g.n2 = n2; g.n3 = n3;
        g.coords = ModifiedKerrSchild(hslope);
        g.x1min = std::log(rIn);
        g.x1max = std::log(rOut);
        g.x2min = 0.0; g.x2max = 1.0;
        g.x3min = 0.0; g.x3max = 2.0 * kPi;
        g.dx1 = (g.x1max - g.x1min) / n1;
        g.dx2 = (g.x2max - g.x2min) / n2;
        g.dx3 = (g.x3max - g.x3min) / n3;
        return g;
    }
};

}  // namespace grmhd
