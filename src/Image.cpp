#include "Image.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace bhsim {

namespace {

// Encodage final d'un canal display-référencé ∈ [0,1] : correction gamma 2.2
// (espace linéaire de calcul -> espace sRGB de l'écran) puis quantification 8 bits.
uint8_t encode(double c) {
    c = std::clamp(c, 0.0, 1.0);
    return static_cast<uint8_t>(std::lround(std::pow(c, 1.0 / 2.2) * 255.0));
}

// Tone-mapping ACES filmic *complet* (fit de Stephen Hill / Narkowicz 2016) :
// l'image HDR sRGB-linéaire est transformée dans l'espace ACEScg via la matrice
// d'entrée RRT, passée dans la courbe RRT∘ODT, puis ramenée en sRGB par la matrice
// de sortie ODT. Contrairement à l'approximation mono-canal, ce pipeline matriciel
// préserve la teinte des très hautes lumières : le côté gauche (beaming Doppler
// intense) révèle la texture du plasma au lieu de saturer en un bloc blanc « brûlé ».
struct RGB { double r, g, b; };

RGB matMul(const double m[3][3], const RGB& v) {
    return { m[0][0]*v.r + m[0][1]*v.g + m[0][2]*v.b,
             m[1][0]*v.r + m[1][1]*v.g + m[1][2]*v.b,
             m[2][0]*v.r + m[2][1]*v.g + m[2][2]*v.b };
}

double rrtOdtFit(double v) {
    const double a = v * (v + 0.0245786) - 0.000090537;
    const double b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

RGB acesFitted(RGB c) {
    static const double IN[3][3] = {
        {0.59719, 0.35458, 0.04823},
        {0.07600, 0.90834, 0.01566},
        {0.02840, 0.13383, 0.83777}};
    static const double OUT[3][3] = {
        { 1.60475, -0.53108, -0.07367},
        {-0.10208,  1.10813, -0.00605},
        {-0.00327, -0.07276,  1.07602}};
    c = matMul(IN, c);
    c.r = rrtOdtFit(c.r); c.g = rrtOdtFit(c.g); c.b = rrtOdtFit(c.b);
    c = matMul(OUT, c);
    c.r = std::clamp(c.r, 0.0, 1.0);
    c.g = std::clamp(c.g, 0.0, 1.0);
    c.b = std::clamp(c.b, 0.0, 1.0);
    return c;
}

void putLE32(std::ofstream& os, uint32_t v) {
    char b[4] = { char(v & 0xFF), char((v >> 8) & 0xFF),
                  char((v >> 16) & 0xFF), char((v >> 24) & 0xFF) };
    os.write(b, 4);
}
void putLE16(std::ofstream& os, uint16_t v) {
    char b[2] = { char(v & 0xFF), char((v >> 8) & 0xFF) };
    os.write(b, 2);
}

}  // namespace

Image::Image(int width, int height)
    : width_(width), height_(height),
      data_(static_cast<size_t>(width) * height * 3, 0.0) {}

void Image::setPixel(int i, int j, double r, double g, double b) {
    const size_t idx = (static_cast<size_t>(j) * width_ + i) * 3;
    data_[idx + 0] = r;
    data_[idx + 1] = g;
    data_[idx + 2] = b;
}

void Image::resolve(int i, int j, uint8_t& r, uint8_t& g, uint8_t& b) const {
    const size_t idx = (static_cast<size_t>(j) * width_ + i) * 3;
    const double lr = data_[idx + 0], lg = data_[idx + 1], lb = data_[idx + 2];

    // HDR linéaire -> ACES filmic complet (matriciel) -> gamma 2.2 -> 8 bits sRGB.
    const RGB tm = acesFitted({ lr, lg, lb });
    r = encode(tm.r);
    g = encode(tm.g);
    b = encode(tm.b);
}

bool Image::writePPM(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    os << "P6\n" << width_ << ' ' << height_ << "\n255\n";
    for (int j = 0; j < height_; ++j) {
        for (int i = 0; i < width_; ++i) {
            uint8_t r, g, b;
            resolve(i, j, r, g, b);
            const char px[3] = { char(r), char(g), char(b) };
            os.write(px, 3);
        }
    }
    return static_cast<bool>(os);
}

bool Image::writeBMP(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;

    const int rowSize  = (width_ * 3 + 3) & ~3;          // lignes alignées sur 4 octets
    const uint32_t dataSize = static_cast<uint32_t>(rowSize) * height_;
    const uint32_t fileSize = 54 + dataSize;

    // BITMAPFILEHEADER (14 octets)
    os.put('B'); os.put('M');
    putLE32(os, fileSize);
    putLE32(os, 0);
    putLE32(os, 54);                                     // offset des données
    // BITMAPINFOHEADER (40 octets)
    putLE32(os, 40);
    putLE32(os, static_cast<uint32_t>(width_));
    putLE32(os, static_cast<uint32_t>(height_));         // positif = lignes bas->haut
    putLE16(os, 1);                                      // plans
    putLE16(os, 24);                                     // bits par pixel
    putLE32(os, 0);                                      // compression : aucune
    putLE32(os, dataSize);
    putLE32(os, 2835); putLE32(os, 2835);               // ~72 DPI
    putLE32(os, 0); putLE32(os, 0);

    const int pad = rowSize - width_ * 3;
    const char zero[3] = { 0, 0, 0 };
    for (int j = height_ - 1; j >= 0; --j) {            // BMP : de bas en haut
        for (int i = 0; i < width_; ++i) {
            uint8_t r, g, b;
            resolve(i, j, r, g, b);
            const char px[3] = { char(b), char(g), char(r) };   // ordre BGR
            os.write(px, 3);
        }
        if (pad) os.write(zero, pad);
    }
    return static_cast<bool>(os);
}

// -----------------------------------------------------------------------------

bool writeMatrixCSV(const std::string& path,
                    const std::vector<double>& data, int width, int height) {
    std::ofstream os(path);
    if (!os) return false;
    os << std::setprecision(8);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            os << data[static_cast<size_t>(j) * width + i];
            if (i + 1 < width) os << ',';
        }
        os << '\n';
    }
    return static_cast<bool>(os);
}

bool writeMatrixCSV(const std::string& path,
                    const std::vector<int>& data, int width, int height) {
    std::ofstream os(path);
    if (!os) return false;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            os << data[static_cast<size_t>(j) * width + i];
            if (i + 1 < width) os << ',';
        }
        os << '\n';
    }
    return static_cast<bool>(os);
}

}  // namespace bhsim
