#pragma once
#include <cmath>
#include <iosfwd>

namespace bhsim {

// -----------------------------------------------------------------------------
// Vecteur 3D (double précision). Algèbre vectorielle inline pour la performance :
// ces opérations sont appelées des milliards de fois dans la boucle de rendu.
// -----------------------------------------------------------------------------
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    constexpr Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s)      { x *= s;   y *= s;   z *= s;   return *this; }

    constexpr double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    constexpr Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double norm2() const { return x * x + y * y + z * z; }
    double norm()  const { return std::sqrt(norm2()); }
    Vec3 normalized() const { double n = norm(); return n > 0.0 ? (*this) / n : *this; }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// Interpolation linéaire (utilisée pour les dégradés et les croisements de disque).
inline Vec3 lerp(const Vec3& a, const Vec3& b, double t) { return a + (b - a) * t; }

std::ostream& operator<<(std::ostream& os, const Vec3& v);

}  // namespace bhsim
