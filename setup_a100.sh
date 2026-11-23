#!/usr/bin/env bash
# Build and run a quick CUDA smoke test on an NVIDIA A100 environment.
# Intended for a CUDA "devel" Ubuntu image on a GPU host.
set -euo pipefail

echo "=== 1/4: dependencies ==="
apt-get update -qq
apt-get install -y -qq cmake build-essential

echo "=== 2/4: GPU / CUDA check ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo "-> An A100 should report compute capability 8.0."
echo "-> Adjust CUDA_ARCHITECTURES in CMakeLists.txt for another GPU target."

echo "=== 3/4: Release build (CUDA auto-detected) ==="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "=== 4/4: quick GPU render (480x270) ==="
./build/blackholesim 480 270 1 0.9 output

echo
echo "OK -- build and smoke render completed."
echo "Check the program output above and confirm that the selected backend is GPU."
