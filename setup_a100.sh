#!/usr/bin/env bash
# Setup + build + test rapide sur un pod GPU A100 (RunPod/Vast.ai, image
# nvidia/cuda:*-devel-ubuntu*). A lancer une fois le repo cloné sur le pod :
#   bash setup_a100.sh
set -euo pipefail

echo "=== 1/4 : dependances ==="
apt-get update -qq
apt-get install -y -qq cmake build-essential

echo "=== 2/4 : verification GPU / CUDA ==="
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv
echo "-> compute_cap doit etre 8.0 pour une A100 (sinon corriger CUDA_ARCHITECTURES dans CMakeLists.txt)"

echo "=== 3/4 : compilation (Release, CUDA auto-detecte) ==="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "=== 4/4 : rendu test rapide (480x270, doit utiliser le backend GPU) ==="
./build/blackholesim 480 270 1 0.9 output

echo ""
echo "OK -- build + rendu GPU fonctionnels. Verifie ci-dessus la ligne 'Backend'"
echo "dans la sortie du programme : elle doit afficher GPU, pas CPU."
