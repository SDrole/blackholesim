#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "Config.hpp"
#include "Metric.hpp"
#include "Renderer.hpp"
#include "Image.hpp"

using namespace bhsim;

namespace {

// Render dispatch: CUDA when compiled in, otherwise the CPU path.
RenderResult renderDispatch(const Renderer& renderer) {
#ifdef BHSIM_CUDA
    return renderer.renderGPU();
#else
    return renderer.render();
#endif
}

// Radius of the Page-Thorne flux peak, used as the animation rotation reference.
double diskPeakRadius(const Kerr& bh, const Config& cfg) {
    const double rin = bh.iscoPrograde();
    double best = rin * 1.5, fmax = -1.0;
    const int N = 2000;
    for (int k = 0; k <= N; ++k) {
        const double r = rin * 1.0001 + (cfg.diskOuter - rin * 1.0001) * k / N;
        const double f = bh.pageThorneFlux(r);
        if (f > fmax) { fmax = f; best = r; }
    }
    return best;
}

// Animation mode: frame_NNN.bmp sequence.
// The disk evolves at the relativistic Keplerian rate while the camera drifts
// in azimuth. The resulting frames can be assembled with ffmpeg.
int runAnimation(Config cfg, const Kerr& bh) {
    const std::filesystem::path dir = std::filesystem::path(cfg.outputDir) / "frames";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const int frames = std::max(1, cfg.animFrames);
    const double rPeak = diskPeakRadius(bh, cfg);
    const double OmegaPeak = bh.orbitalOmega(rPeak);
    const double Ttotal = cfg.animDiskTurns * 2.0 * kPi / OmegaPeak;
    const double azTotal = cfg.animCameraOrbitDeg * kPi / 180.0;

    std::cout << "  Animation mode: " << frames << " frames\n"
              << "  Peak radius r*  = " << rPeak << "  ->  Omega(r*) = " << OmegaPeak << "\n"
              << "  Sequence time   = " << Ttotal << " M  (" << cfg.animDiskTurns
              << " Keplerian turn(s) at the peak)\n"
              << "  Camera orbit    = " << cfg.animCameraOrbitDeg << " deg\n"
              << "--------------------------------------------------\n";

    Renderer renderer(cfg);

    const auto t0 = std::chrono::steady_clock::now();
    double maxDrift = 0.0;
    for (int f = 0; f < frames; ++f) {
        const double u = static_cast<double>(f) / frames;
        renderer.setFrame(Ttotal * u, azTotal * u);

        const RenderResult res = renderDispatch(renderer);
        maxDrift = std::max(maxDrift, res.maxCarterDrift);

        char name[32];
        std::snprintf(name, sizeof(name), "frame_%03d.bmp", f + 1);
        const auto path = (dir / name).string();
        if (res.image.writeBMP(path))
            std::cout << "  [" << (f + 1) << "/" << frames << "] -> " << path << "\n";
        else
            std::cerr << "  Failed to write " << path << "\n";
    }
    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "--------------------------------------------------\n"
              << "  Sequence completed in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n"
              << "  Maximum Carter drift over sequence: " << maxDrift << "\n"
              << "  Example video assembly command:\n"
              << "    ffmpeg -framerate 30 -i " << (dir / "frame_%03d.bmp").string()
              << " -c:v libx264 -pix_fmt yuv420p blackhole.mp4\n";
    return 0;
}

void printBanner(const Config& cfg, const Kerr& bh) {
    std::cout << "==================================================\n"
              << "  Kerr black hole simulator\n"
              << "  null geodesics, canonical Hamiltonian, G = c = 1\n"
              << "==================================================\n"
              << "  Mass M             : " << bh.mass() << "\n"
              << "  Spin a/M           : " << bh.spinStar()
              << "   (a = " << bh.spin() << ")\n"
              << "  Outer horizon r+   : " << bh.horizonOuter() << "\n"
              << "  Inner horizon r-   : " << bh.horizonInner() << "\n"
              << "  Equatorial ergosphere: " << bh.ergosphereEquator() << "\n"
              << "  Prograde ISCO      : " << bh.iscoPrograde() << "\n"
              << "  Image              : " << cfg.width << " x " << cfg.height
              << "   (" << cfg.samples << "x" << cfg.samples << " AA)\n"
              << "  N.-Thorne disk     : r in [" << bh.iscoPrograde() << ", "
              << cfg.diskOuter << "]\n"
#ifdef BHSIM_CUDA
              << "  Backend            : GPU (CUDA, parallel RKF45)\n"
#else
              << "  Backend            : CPU\n"
#endif
              << "--------------------------------------------------\n";
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;

    // Animation:
    //   blackholesim anim [frames] [spin a/M] [output_dir] [width] [height] [AA]
    // Still image:
    //   blackholesim [width] [height] [AA] [spin a/M] [output_dir]
    if (argc >= 2 && std::strcmp(argv[1], "anim") == 0) {
        cfg.animate = true;
        if (argc >= 3) cfg.animFrames = std::max(1, std::atoi(argv[2]));
        if (argc >= 4) cfg.spin       = std::clamp(std::atof(argv[3]), 0.0, 0.998);
        if (argc >= 5) cfg.outputDir  = argv[4];
        if (argc >= 6) cfg.width      = std::max(1, std::atoi(argv[5]));
        if (argc >= 7) cfg.height     = std::max(1, std::atoi(argv[6]));
        if (argc >= 8) cfg.samples    = std::max(1, std::atoi(argv[7]));

        Kerr bh(cfg.M, cfg.M * cfg.spin);
        printBanner(cfg, bh);
        return runAnimation(cfg, bh);
    }

    if (argc >= 2) cfg.width   = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) cfg.height  = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) cfg.samples = std::max(1, std::atoi(argv[3]));
    if (argc >= 5) cfg.spin    = std::clamp(std::atof(argv[4]), 0.0, 0.998);
    if (argc >= 6) cfg.outputDir = argv[5];

    Kerr bh(cfg.M, cfg.M * cfg.spin);
    printBanner(cfg, bh);

    std::error_code ec;
    std::filesystem::create_directories(cfg.outputDir, ec);

    Renderer renderer(cfg);

    const auto t0 = std::chrono::steady_clock::now();
    const RenderResult res = renderDispatch(renderer);
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "  Render completed in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // Carter-constant validation.
    const bool carterOK = res.maxCarterDrift < cfg.carterTolerance;
    std::cout << "  Carter constant K: max drift = " << res.maxCarterDrift
              << "  (tol " << cfg.carterTolerance << ")  ->  "
              << (carterOK ? "CONSERVED [OK]" : "DRIFT [!]") << "\n"
              << "--------------------------------------------------\n";

    const std::filesystem::path dir(cfg.outputDir);
    int written = 0;

    if (cfg.writeImage) {
        const auto bmp = (dir / "blackhole_kerr.bmp").string();
        const auto ppm = (dir / "blackhole_kerr.ppm").string();
        if (res.image.writeBMP(bmp)) { std::cout << "  -> " << bmp << "\n"; ++written; }
        if (res.image.writePPM(ppm)) { std::cout << "  -> " << ppm << "\n"; ++written; }
    }

    if (cfg.writeMatrices) {
        const auto fRed = (dir / "redshift.csv").string();
        const auto fHit = (dir / "hit_type.csv").string();
        const auto fCar = (dir / "carter_drift.csv").string();
        if (writeMatrixCSV(fRed, res.redshift, res.width, res.height))
            { std::cout << "  -> " << fRed << "  (disk redshift factor g)\n"; ++written; }
        if (writeMatrixCSV(fHit, res.hitType, res.width, res.height))
            { std::cout << "  -> " << fHit << "  (0=background 1=disk 2=horizon)\n"; ++written; }
        if (writeMatrixCSV(fCar, res.carterDrift, res.width, res.height))
            { std::cout << "  -> " << fCar << "  (relative Carter drift)\n"; ++written; }
    }

    std::cout << "--------------------------------------------------\n"
              << "  " << written << " file(s) written to '" << cfg.outputDir << "'.\n";
    return written > 0 ? 0 : 1;
}
