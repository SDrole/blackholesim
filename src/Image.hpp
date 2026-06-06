#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bhsim {

// -----------------------------------------------------------------------------
// Tampon image RGB en virgule flottante (linéaire, HDR). L'écriture applique le
// tone-mapping ACES filmic puis l'encodage gamma 2.2 (sRGB).
//   - PPM (P6) : format graphique standard, sans dépendance.
//   - BMP (24 bits) : s'ouvre nativement sous Windows.
// -----------------------------------------------------------------------------
class Image {
public:
    Image() = default;
    Image(int width, int height);

    int width()  const { return width_; }
    int height() const { return height_; }

    void setPixel(int i, int j, double r, double g, double b);

    bool writePPM(const std::string& path) const;
    bool writeBMP(const std::string& path) const;

private:
    // Résout un pixel linéaire HDR en RGB 8 bits affichables (tone-map + gamma).
    void resolve(int i, int j, uint8_t& r, uint8_t& g, uint8_t& b) const;

    int width_ = 0, height_ = 0;
    std::vector<double> data_;        // 3 * width * height, RGB linéaire
};

// --- Export de matrices scientifiques au format CSV (lignes j, colonnes i) ---
bool writeMatrixCSV(const std::string& path,
                    const std::vector<double>& data, int width, int height);
bool writeMatrixCSV(const std::string& path,
                    const std::vector<int>& data, int width, int height);

}  // namespace bhsim
