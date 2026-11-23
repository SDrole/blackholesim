#include "Hdf5Writer.hpp"

#ifdef BHSIM_HAVE_HDF5
#include <hdf5.h>
#endif
#include <vector>

namespace grmhd {

#ifdef BHSIM_HAVE_HDF5

namespace {

bool writeDataset1D(hid_t loc, const char* name, const std::vector<double>& v) {
    const hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
    const hid_t space = H5Screate_simple(1, dims, nullptr);
    if (space < 0) return false;
    const hid_t dset = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); return false; }
    const herr_t st = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, v.data());
    H5Dclose(dset);
    H5Sclose(space);
    return st >= 0;
}

bool writeScalarAttr(hid_t loc, const char* name, double value) {
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                                  H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) { H5Sclose(space); return false; }
    const herr_t st = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr); H5Sclose(space);
    return st >= 0;
}

bool writeIntAttr(hid_t loc, const char* name, int value) {
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT, space,
                                  H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) { H5Sclose(space); return false; }
    const herr_t st = H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr); H5Sclose(space);
    return st >= 0;
}

bool writeStringAttr(hid_t loc, const char* name, const char* value) {
    const hid_t stype = H5Tcopy(H5T_C_S1);
    H5Tset_size(stype, H5T_VARIABLE);
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name, stype, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) { H5Sclose(space); H5Tclose(stype); return false; }
    const herr_t st = H5Awrite(attr, stype, &value);
    H5Aclose(attr); H5Sclose(space); H5Tclose(stype);
    return st >= 0;
}

bool readDataset1D(hid_t loc, const char* name, std::vector<double>& v) {
    const hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    if (dset < 0) return false;
    const hid_t space = H5Dget_space(dset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    v.resize(static_cast<size_t>(dims[0]));
    const herr_t st = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, v.data());
    H5Sclose(space); H5Dclose(dset);
    return st >= 0;
}

}  // namespace

bool writeSnapshotHDF5(const Solver1D& solver, const std::string& path) {
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return false;

    const int NG = Solver1D::NG;
    const int nx = solver.nx();
    std::vector<double> x(nx), rho(nx), uu(nx), u1(nx), u2(nx), u3(nx);
    for (int i = 0; i < nx; ++i) {
        const PrimitiveState& P = solver.primitives()[NG + i];
        x[i]   = solver.xc(NG + i);
        rho[i] = P.rho;
        uu[i]  = P.u;
        u1[i]  = P.utilde[0];
        u2[i]  = P.utilde[1];
        u3[i]  = P.utilde[2];
    }

    bool ok = true;
    const hid_t prims = H5Gcreate2(file, "prims", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (prims < 0) { H5Fclose(file); return false; }

    // Dataset names follow the iharm convention.
    ok = ok && writeDataset1D(prims, "RHO", rho);
    ok = ok && writeDataset1D(prims, "UU",  uu);
    ok = ok && writeDataset1D(prims, "U1",  u1);
    ok = ok && writeDataset1D(prims, "U2",  u2);
    ok = ok && writeDataset1D(prims, "U3",  u3);
    H5Gclose(prims);

    const hid_t grid = H5Gcreate2(file, "grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ok = ok && (grid >= 0);
    if (grid >= 0) { ok = ok && writeDataset1D(grid, "x", x); H5Gclose(grid); }

    ok = ok && writeScalarAttr(file, "t", solver.time());
    ok = ok && writeIntAttr(file, "nx", nx);
    ok = ok && writeScalarAttr(file, "gam", solver.config().gamma);

    // Provenance markers: this file is explicitly NOT a full GRMHD snapshot.
    ok = ok && writeStringAttr(file, "bhsim_stage", "S3-SRHD-flat");
    ok = ok && writeStringAttr(file, "metric", "minkowski-cartesian-1d");
    ok = ok && writeStringAttr(file, "absent_fields",
                               "B1,B2,B3 (no magnetic field before S5); "
                               "a,lapse,shift,FMKS grid (flat space, no curved metric)");
    H5Fclose(file);
    return ok;
}

namespace {

bool readStringAttr(hid_t loc, const char* name, std::string& out) {
    const hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return false;
    const hid_t stype = H5Tcopy(H5T_C_S1);
    H5Tset_size(stype, H5T_VARIABLE);
    char* buf = nullptr;
    const herr_t st = H5Aread(attr, stype, &buf);
    if (st >= 0 && buf) { out = buf; H5free_memory(buf); }
    H5Tclose(stype); H5Aclose(attr);
    return st >= 0;
}

}  // namespace

bool readSnapshotHDF5(const std::string& path, SnapshotContents& out) {
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return false;

    bool ok = true;
    const hid_t prims = H5Gopen2(file, "prims", H5P_DEFAULT);
    if (prims < 0) { H5Fclose(file); return false; }

    // Read back every dataset that was written. Otherwise the round trip does
    // not test fields the reader forgot about.
    ok = ok && readDataset1D(prims, "RHO", out.rho);
    ok = ok && readDataset1D(prims, "UU",  out.uu);
    ok = ok && readDataset1D(prims, "U1",  out.u1);
    ok = ok && readDataset1D(prims, "U2",  out.u2);
    ok = ok && readDataset1D(prims, "U3",  out.u3);
    H5Gclose(prims);

    const hid_t grid = H5Gopen2(file, "grid", H5P_DEFAULT);
    if (grid >= 0) { ok = ok && readDataset1D(grid, "x", out.x); H5Gclose(grid); }
    else ok = false;

    hid_t attr = H5Aopen(file, "t", H5P_DEFAULT);
    if (attr >= 0) { H5Aread(attr, H5T_NATIVE_DOUBLE, &out.t); H5Aclose(attr); } else ok = false;
    attr = H5Aopen(file, "gam", H5P_DEFAULT);
    if (attr >= 0) { H5Aread(attr, H5T_NATIVE_DOUBLE, &out.gam); H5Aclose(attr); } else ok = false;
    attr = H5Aopen(file, "nx", H5P_DEFAULT);
    if (attr >= 0) { H5Aread(attr, H5T_NATIVE_INT, &out.nx); H5Aclose(attr); } else ok = false;

    // Structural attributes carry the "this is not a complete iharm snapshot"
    // contract, so they must survive the round trip too.
    ok = ok && readStringAttr(file, "bhsim_stage", out.stage);
    ok = ok && readStringAttr(file, "metric", out.metric);
    ok = ok && readStringAttr(file, "absent_fields", out.absentFields);

    H5Fclose(file);
    return ok;
}

#else   // BHSIM_HAVE_HDF5 not defined

// HDF5 is unavailable in this build. Expose explicit failing stubs rather than
// pretending that anything was serialized.
bool writeSnapshotHDF5(const Solver1D&, const std::string&) { return false; }
bool readSnapshotHDF5(const std::string&, SnapshotContents&) { return false; }

#endif  // BHSIM_HAVE_HDF5

}  // namespace grmhd
