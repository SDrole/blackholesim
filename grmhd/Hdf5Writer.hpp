#pragma once
#include <string>
#include "Solver1D.hpp"

namespace grmhd {

// -----------------------------------------------------------------------------
// Minimal HDF5 writer introduced during S3.
//
// At this stage the goal is to validate serialization infrastructure, not to
// pretend that an SRHD state is a complete GRMHD snapshot.
//
// iharm/HARM-oriented schema:
//   prims/RHO, prims/UU, prims/U1..U3
//
// Fields intentionally absent at S3:
//   * prims/B1..B3 — no magnetic field before S5.
//   * curved-metric/FMKS metadata, spin, lapse, shift — S3 is flat spacetime.
//   * t is present, but it is Minkowski coordinate time.
//
// The bhsim_stage attribute is set to "S3-SRHD-flat" so readers cannot mistake
// this file for a full GRMHD snapshot.
// -----------------------------------------------------------------------------

// Write the current solver state. Returns false on HDF5 failure.
bool writeSnapshotHDF5(const Solver1D& solver, const std::string& path);

// Round-trip structure covering every dataset and attribute actually written.
// A reader test that ignores fields does not validate those fields.
struct SnapshotContents {
    std::vector<double> x, rho, uu, u1, u2, u3;
    double t = 0.0;
    double gam = 0.0;
    int nx = 0;
    std::string stage;          // bhsim_stage
    std::string metric;         // metric
    std::string absentFields;   // absent_fields
};

bool readSnapshotHDF5(const std::string& path, SnapshotContents& out);

}  // namespace grmhd
