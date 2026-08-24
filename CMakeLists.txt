cmake_minimum_required(VERSION 3.18)
project(blackholesim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Release par défaut (optimisations indispensables pour le rendu).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- Backend GPU (CUDA) ------------------------------------------------------
# Activé automatiquement si un compilateur CUDA (nvcc) est disponible. Sur une
# machine sans CUDA (ex. poste Windows sans toolkit), le rendu reste CPU pur.
option(BHSIM_CUDA "Construire le moteur de rendu GPU (CUDA) si nvcc est présent" ON)
include(CheckLanguage)
set(BHSIM_USE_CUDA OFF)
if(BHSIM_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(BHSIM_USE_CUDA ON)
    message(STATUS "CUDA activé : ${CMAKE_CUDA_COMPILER}")
  else()
    message(STATUS "CUDA demandé mais nvcc introuvable : rendu CPU uniquement.")
  endif()
endif()

set(BHSIM_SOURCES
  src/main.cpp
  src/Vec3.cpp
  src/Metric.cpp
  src/Geodesic.cpp
  src/Integrator.cpp
  src/Camera.cpp
  src/Renderer.cpp
  src/Image.cpp
)
if(BHSIM_USE_CUDA)
  list(APPEND BHSIM_SOURCES src/cuda/Renderer.cu)
endif()

add_executable(blackholesim ${BHSIM_SOURCES})
target_include_directories(blackholesim PRIVATE src)

if(BHSIM_USE_CUDA)
  target_compile_definitions(blackholesim PRIVATE BHSIM_CUDA)
  # A100 (PCIe ou SXM, même puce GA100) = Ampere = sm_80. (Adapter pour un autre GPU :
  # RTX 3090->86, RTX 4090->89, H100->90.)
  set_target_properties(blackholesim PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  # Lambdas device tolérées ; PAS de -use_fast_math (le FP64 exact préserve Carter).
  target_compile_options(blackholesim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# Avertissements + maths rapides : appliqués UNIQUEMENT au C++ (pas au device CUDA,
# que l'on garde en arithmétique IEEE exacte). Le niveau d'optimisation (/O2, -O3)
# vient du type de build (Release).
if(MSVC)
  target_compile_options(blackholesim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /fp:fast /EHsc /arch:AVX2>)
else()
  target_compile_options(blackholesim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -ffast-math -mavx2 -mfma>)
endif()

# --- Solveur GRMHD (exécutable SÉPARÉ du ray tracer) -------------------------
# Architecture volontairement découplée :
#   grmhd_tests / solveur  ->  snapshots  ->  blackholesim (ray tracer)
# Le solveur ne connaît pas le moteur de rendu et réciproquement : chacun est
# validable indépendamment. Cf. docs/ROADMAP_5_AXES.md (axe 5).
add_executable(grmhd_tests
  grmhd/KerrSchild.cpp
  grmhd/tests_s1.cpp
)
target_include_directories(grmhd_tests PRIVATE grmhd)
set_target_properties(grmhd_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  # PAS de /fp:fast ici : les tests de validation mesurent des écarts au
  # niveau de la précision machine, les optimisations flottantes agressives
  # fausseraient le verdict.
  target_compile_options(grmhd_tests PRIVATE /W4 /EHsc)
else()
  target_compile_options(grmhd_tests PRIVATE -Wall -Wextra)
endif()

# Consolidation S1 : vérification du schéma de dérivation (pas complexe) et
# validation croisée contre le moteur Boyer-Lindquist du ray tracer, utilisé
# comme ORACLE INDÉPENDANT (déjà validé en production par la conservation de
# Carter). D'où le lien avec src/Metric.cpp et src/Geodesic.cpp.
add_executable(grmhd_s1_consolidation
  grmhd/KerrSchild.cpp
  grmhd/tests_s1_consolidation.cpp
  src/Metric.cpp
  src/Geodesic.cpp
)
target_include_directories(grmhd_s1_consolidation PRIVATE grmhd src)
set_target_properties(grmhd_s1_consolidation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_options(grmhd_s1_consolidation PRIVATE /W4 /EHsc)
else()
  target_compile_options(grmhd_s1_consolidation PRIVATE -Wall -Wextra)
endif()

# Parallélisation du rendu CPU (OpenMP) — optionnelle.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(blackholesim PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP activé (rendu CPU multi-thread).")
else()
  message(STATUS "OpenMP non trouvé : rendu CPU mono-thread.")
endif()
