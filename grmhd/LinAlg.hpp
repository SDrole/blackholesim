#pragma once
#include <cmath>

namespace grmhd {

// -----------------------------------------------------------------------------
// Minimal 4x4 linear algebra for general relativity.
// Deliberately self-contained: these routines run in tight loops and should
// remain easy to inspect and test.
// -----------------------------------------------------------------------------

// Gauss-Jordan 4x4 inversion with partial pivoting.
// Returns false for a singular matrix (pivot below threshold).
inline bool invert4x4(const double m[4][4], double out[4][4]) {
    double a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i][j] = m[i][j];
            a[i][j + 4] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < 4; ++col) {
        // Partial pivoting is important because Kerr-Schild metric components
        // can differ substantially in magnitude near the horizon.
        int piv = col;
        double best = std::fabs(a[col][col]);
        for (int i = col + 1; i < 4; ++i) {
            const double v = std::fabs(a[i][col]);
            if (v > best) { best = v; piv = i; }
        }
        if (best < 1e-300) return false;
        if (piv != col) {
            for (int j = 0; j < 8; ++j) {
                const double t = a[col][j]; a[col][j] = a[piv][j]; a[piv][j] = t;
            }
        }

        const double inv = 1.0 / a[col][col];
        for (int j = 0; j < 8; ++j) a[col][j] *= inv;

        for (int i = 0; i < 4; ++i) {
            if (i == col) continue;
            const double f = a[i][col];
            if (f == 0.0) continue;
            for (int j = 0; j < 8; ++j) a[i][j] -= f * a[col][j];
        }
    }

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out[i][j] = a[i][j + 4];
    return true;
}

// 4x4 determinant via 2x2 cofactor expansion; allocation-free.
inline double det4x4(const double m[4][4]) {
    const double s0 = m[0][0]*m[1][1] - m[1][0]*m[0][1];
    const double s1 = m[0][0]*m[1][2] - m[1][0]*m[0][2];
    const double s2 = m[0][0]*m[1][3] - m[1][0]*m[0][3];
    const double s3 = m[0][1]*m[1][2] - m[1][1]*m[0][2];
    const double s4 = m[0][1]*m[1][3] - m[1][1]*m[0][3];
    const double s5 = m[0][2]*m[1][3] - m[1][2]*m[0][3];

    const double c5 = m[2][2]*m[3][3] - m[3][2]*m[2][3];
    const double c4 = m[2][1]*m[3][3] - m[3][1]*m[2][3];
    const double c3 = m[2][1]*m[3][2] - m[3][1]*m[2][2];
    const double c2 = m[2][0]*m[3][3] - m[3][0]*m[2][3];
    const double c1 = m[2][0]*m[3][2] - m[3][0]*m[2][2];
    const double c0 = m[2][0]*m[3][1] - m[3][0]*m[2][1];

    return s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
}

}  // namespace grmhd
