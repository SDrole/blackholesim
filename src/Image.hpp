#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bhsim {

class Image {
public:
    Image() = default;
    Image(int width, int height);

    int width()  const { return width_; }
    int height() const { return height_; }

    void setPixel(int i, int j, double r, double g, double b);

    void applyPointSpreadFunction(double haloRadiusFrac, double haloWeight);

    bool writePPM(const std::string& path) const;
    bool writeBMP(const std::string& path) const;

private:
    void resolve(int i, int j, uint8_t& r, uint8_t& g, uint8_t& b) const;

    int width_ = 0, height_ = 0;
    std::vector<double> data_;
};

// --- Export de matrices scientifiques au format CSV (lignes j, colonnes i) ---
bool writeMatrixCSV(const std::string& path,
                    const std::vector<double>& data, int width, int height);
bool writeMatrixCSV(const std::string& path,
                    const std::vector<int>& data, int width, int height);

}  // namespace bhsim
