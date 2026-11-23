cmake_minimum_required(VERSION 3.18)
project(blackholesim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BHSIM_CUDA "Build the CUDA renderer when nvcc is available" ON)
include(CheckLanguage)
set(BHSIM_USE_CUDA OFF)
if(BHSIM_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(BHSIM_USE_CUDA ON)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
  else()
    message(STATUS "CUDA requested but nvcc was not found: CPU renderer only.")
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
  set_target_properties(blackholesim PROPERTIES
    CUDA_ARCHITECTURES "80"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  target_compile_options(blackholesim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

if(MSVC)
  target_compile_options(blackholesim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /fp:fast /EHsc /arch:AVX2>)
else()
  target_compile_options(blackholesim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -ffast-math -mavx2 -mfma>)
endif()

add_executable(grmhd_tests
  grmhd/KerrSchild.cpp
  grmhd/tests_s1.cpp
)
target_include_directories(grmhd_tests PRIVATE grmhd)
set_target_properties(grmhd_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_options(grmhd_tests PRIVATE /W4 /EHsc)
else()
  target_compile_options(grmhd_tests PRIVATE -Wall -Wextra)
endif()

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
    message(STATUS "HDF5 found via environment: ${BHSIM_HDF5_LIBRARY}")
  endif()
endif()

if(HDF5_FOUND)
  message(STATUS "HDF5 enabled: S3 serialization tests included.")
else()
  message(WARNING "HDF5 not found: S3 serialization tests will be skipped. "
                  "Pass -DHDF5_ROOT=<prefix> to enable them.")
endif()

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

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(blackholesim PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled (multi-threaded CPU renderer).")
else()
  message(STATUS "OpenMP not found: single-threaded CPU renderer.")
endif()
