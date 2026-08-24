#include "Vec3.hpp"
#include <ostream>

namespace bhsim {

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
    return os;
}

}  // namespace bhsim
