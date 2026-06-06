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

// Dispatch du rendu : GPU (CUDA) si disponible à la compilation, sinon CPU.
RenderResult renderDispatch(const Renderer& renderer) {
#ifdef BHSIM_CUDA
    return renderer.renderGPU();
#else
    return renderer.render();
#endif
}

// Rayon du pic du flux de Page-Thorne (référence pour la vitesse de rotation).
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

// --- Mode animation : séquence d'images frame_NNN.bmp ------------------------
// Le disque tourne à la vitesse képlérienne relativiste (via la phase t) et la
// caméra dérive légèrement sur son orbite ZAMO (azimut). Boucle exploitable en
// vidéo (ffmpeg -i frame_%03d.bmp ... ).
int runAnimation(Config cfg, const Kerr& bh) {
    const std::filesystem::path dir = std::filesystem::path(cfg.outputDir) / "frames";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const int frames = std::max(1, cfg.animFrames);
    const double rPeak = diskPeakRadius(bh, cfg);
    const double OmegaPeak = bh.orbitalOmega(rPeak);
    const double Ttotal = cfg.animDiskTurns * 2.0 * kPi / OmegaPeak;   // durée (en M)
    const double azTotal = cfg.animCameraOrbitDeg * kPi / 180.0;

    std::cout << "  Mode ANIMATION : " << frames << " images\n"
              << "  Rayon de pic r* = " << rPeak << "  ->  Ω(r*) = " << OmegaPeak << "\n"
              << "  Durée séquence  = " << Ttotal << " M  (" << cfg.animDiskTurns
              << " tour(s) képlérien(s) au pic)\n"
              << "  Orbite caméra   = " << cfg.animCameraOrbitDeg << " deg\n"
              << "--------------------------------------------------\n";

    Renderer renderer(cfg);   // LUT Planck + texture ciel partagées sur la séquence

    const auto t0 = std::chrono::steady_clock::now();
    double maxDrift = 0.0;
    for (int f = 0; f < frames; ++f) {
        const double u = static_cast<double>(f) / frames;   // [0,1)
        renderer.setFrame(Ttotal * u, azTotal * u);

        const RenderResult res = renderDispatch(renderer);
        maxDrift = std::max(maxDrift, res.maxCarterDrift);

        char name[32];
        std::snprintf(name, sizeof(name), "frame_%03d.bmp", f + 1);
        const auto path = (dir / name).string();
        if (res.image.writeBMP(path))
            std::cout << "  [" << (f + 1) << "/" << frames << "] -> " << path << "\n";
        else
            std::cerr << "  ECHEC ecriture " << path << "\n";
    }
    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "--------------------------------------------------\n"
              << "  Séquence terminée en "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n"
              << "  Dérive max de Carter sur la séquence : " << maxDrift << "\n"
              << "  Assemblage vidéo (exemple) :\n"
              << "    ffmpeg -framerate 30 -i " << (dir / "frame_%03d.bmp").string()
              << " -c:v libx264 -pix_fmt yuv420p blackhole.mp4\n";
    return 0;
}

void printBanner(const Config& cfg, const Kerr& bh) {
    std::cout << "==================================================\n"
              << "  Simulateur de trou noir de Kerr (en rotation)\n"
              << "  géodésiques nulles, hamiltonien canonique, G = c = 1\n"
              << "==================================================\n"
              << "  Masse M            : " << bh.mass() << "\n"
              << "  Spin a/M           : " << bh.spinStar()
              << "   (a = " << bh.spin() << ")\n"
              << "  Horizon externe r+ : " << bh.horizonOuter() << "\n"
              << "  Horizon interne r- : " << bh.horizonInner() << "\n"
              << "  Ergosphère (éq.)   : " << bh.ergosphereEquator() << "\n"
              << "  ISCO (prograde)    : " << bh.iscoPrograde() << "\n"
              << "  Image              : " << cfg.width << " x " << cfg.height
              << "   (" << cfg.samples << "x" << cfg.samples << " AA)\n"
              << "  Disque (N.-Thorne) : r ∈ [" << bh.iscoPrograde() << ", "
              << cfg.diskOuter << "]\n"
#ifdef BHSIM_CUDA
              << "  Backend            : GPU (CUDA, RKF45 parallèle)\n"
#else
              << "  Backend            : CPU" << "\n"
#endif
              << "--------------------------------------------------\n";
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;

    // Mode animation :
    //   blackholesim anim [images] [spin a/M] [dossier_sortie] [largeur] [hauteur] [AA]
    // Image fixe :
    //   blackholesim [largeur] [hauteur] [AA] [spin a/M] [dossier_sortie]
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

    // CLI :  blackholesim [largeur] [hauteur] [AA] [spin a/M] [dossier_sortie]
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
    std::cout << "  Rendu terminé en "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // Vérification de la constante de Carter.
    const bool carterOK = res.maxCarterDrift < cfg.carterTolerance;
    std::cout << "  Constante de Carter K : dérive max = " << res.maxCarterDrift
              << "  (tol " << cfg.carterTolerance << ")  ->  "
              << (carterOK ? "CONSERVÉE [OK]" : "DÉRIVE [!]") << "\n"
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
            { std::cout << "  -> " << fRed << "  (facteur g du disque)\n"; ++written; }
        if (writeMatrixCSV(fHit, res.hitType, res.width, res.height))
            { std::cout << "  -> " << fHit << "  (0=fond 1=disque 2=horizon)\n"; ++written; }
        if (writeMatrixCSV(fCar, res.carterDrift, res.width, res.height))
            { std::cout << "  -> " << fCar << "  (dérive relative de K)\n"; ++written; }
    }

    std::cout << "--------------------------------------------------\n"
              << "  " << written << " fichier(s) écrit(s) dans '" << cfg.outputDir << "'.\n";
    return written > 0 ? 0 : 1;
}
