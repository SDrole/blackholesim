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

# S2 : primitives <-> conservées, hydrodynamique relativiste pure.
add_executable(grmhd_s2_tests
  grmhd/Fluid.cpp
  grmhd/tests_s2.cpp
)
target_include_directories(grmhd_s2_tests PRIVATE grmhd)
set_target_properties(grmhd_s2_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_options(grmhd_s2_tests PRIVATE /W4 /EHsc)
else()
  target_compile_options(grmhd_s2_tests PRIVATE -Wall -Wextra)
endif()

# --- Détection HDF5 PORTABLE -------------------------------------------------
# Aucun chemin utilisateur en dur : le dépôt doit configurer sous Windows local
# comme sur nos machines Linux/A100.
#   1. find_package standard (couvre les installations système Linux, apt/yum,
#      modules HPC, et Windows si HDF5_ROOT est fourni) ;
#   2. à défaut, indices dérivés de l'ENVIRONNEMENT (CONDA_PREFIX, ou le
#      répertoire du python trouvé sur le PATH) — jamais d'un nom d'utilisateur.
#   3. override explicite toujours possible : -DHDF5_ROOT=/chemin
# Si HDF5 reste introuvable, la cible se construit quand même, sans les tests
# de sérialisation (BHSIM_HAVE_HDF5 non défini) : on ne bloque pas tout le
# projet sur une dépendance optionnelle.
find_package(HDF5 QUIET COMPONENTS C)

if(NOT HDF5_FOUND)
  set(_bhsim_hdf5_hints "")
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _bhsim_hdf5_hints "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/Library")
  endif()
  find_program(_bhsim_python NAMES python python3)
  if(_bhsim_python)
    get_filename_component(_bhsim_pydir "${_bhsim_python}" DIRECTORY)
    list(APPEND _bhsim_hdf5_hints "${_bhsim_pydir}" "${_bhsim_pydir}/Library")
    get_filename_component(_bhsim_pyroot "${_bhsim_pydir}" DIRECTORY)
    list(APPEND _bhsim_hdf5_hints "${_bhsim_pyroot}" "${_bhsim_pyroot}/Library")
  endif()

  find_path(BHSIM_HDF5_INCLUDE_DIR hdf5.h
            HINTS ${_bhsim_hdf5_hints} PATH_SUFFIXES include)
  find_library(BHSIM_HDF5_LIBRARY NAMES hdf5 libhdf5
               HINTS ${_bhsim_hdf5_hints} PATH_SUFFIXES lib)
  if(BHSIM_HDF5_INCLUDE_DIR AND BHSIM_HDF5_LIBRARY)
    set(HDF5_FOUND TRUE)
    set(HDF5_INCLUDE_DIRS "${BHSIM_HDF5_INCLUDE_DIR}")
    set(HDF5_C_LIBRARIES  "${BHSIM_HDF5_LIBRARY}")
    message(STATUS "HDF5 trouvé via l'environnement : ${BHSIM_HDF5_LIBRARY}")
  endif()
endif()

if(HDF5_FOUND)
  message(STATUS "HDF5 activé : tests de sérialisation S3 inclus.")
else()
  message(WARNING "HDF5 introuvable : les tests de sérialisation S3 seront ignorés. "
                  "Passer -DHDF5_ROOT=<prefix> pour l'activer.")
endif()

# S3 : solveur SRHD 1D en espace plat + oracle de Riemann exact + HDF5.
add_executable(grmhd_s3_tests
  grmhd/Fluid.cpp
  grmhd/Flux.cpp
  grmhd/Riemann.cpp
  grmhd/Solver1D.cpp
  grmhd/ExactRiemannSRHD.cpp
  grmhd/Hdf5Writer.cpp
  grmhd/tests_s3.cpp
)
target_include_directories(grmhd_s3_tests PRIVATE grmhd)
if(HDF5_FOUND)
  target_include_directories(grmhd_s3_tests PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(grmhd_s3_tests PRIVATE ${HDF5_C_LIBRARIES})
  target_compile_definitions(grmhd_s3_tests PRIVATE BHSIM_HAVE_HDF5)
  # Sur Windows, HDF5 packagé (conda) est une DLL : sans ce define, les
  # variables globales (H5T_NATIVE_DOUBLE_g…) ne sont pas dllimport et le
  # lien échoue. Inoffensif ailleurs.
  if(WIN32)
    target_compile_definitions(grmhd_s3_tests PRIVATE H5_BUILT_AS_DYNAMIC_LIB)
  endif()
endif()
set_target_properties(grmhd_s3_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_options(grmhd_s3_tests PRIVATE /W4 /EHsc /D_USE_MATH_DEFINES)
else()
  target_compile_options(grmhd_s3_tests PRIVATE -Wall -Wextra)
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
