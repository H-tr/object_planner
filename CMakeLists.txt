cmake_minimum_required(VERSION 3.18)
project(ObjectPlanner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ----------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# xsimd: SIMD primitives used by the collision checker's inner loop.
# Header-only; we fetch it so downstream consumers get a vetted version.
include(FetchContent)
FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 8.1.0
)
FetchContent_MakeAvailable(xsimd)

# --- Header-only library --------------------------------------------------
# Downstream C++ consumers do:
#     add_subdirectory(third_party/object_planner)
#     target_link_libraries(my_project PRIVATE object_planner::object_planner)
add_library(object_planner_core INTERFACE)
add_library(object_planner::object_planner ALIAS object_planner_core)
target_include_directories(object_planner_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${xsimd_SOURCE_DIR}/include
)
target_compile_features(object_planner_core INTERFACE cxx_std_17)

install(DIRECTORY include/ DESTINATION include)

# --- Python bindings (nanobind, opt-in) -----------------------------------
# Default ON so scikit-build-core's wheel build does not need an extra
# flag. CI for the C++ tests passes -DOBJECT_PLANNER_BUILD_PYTHON=OFF so
# nanobind is not required there.
option(OBJECT_PLANNER_BUILD_PYTHON "Build the nanobind Python module" ON)
if(OBJECT_PLANNER_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(object_planner_py
    NB_STATIC
    py_src/object_planner_py/py_bindings.cpp
  )
  target_link_libraries(object_planner_py PRIVATE object_planner_core)
  install(TARGETS object_planner_py DESTINATION object_planner_py)
endif()

# --- C++ tests (Catch2, opt-in) -------------------------------------------
# Built via `cmake -DOBJECT_PLANNER_BUILD_TESTS=ON -B build` so the Python
# wheel build (scikit-build-core) does not pull in Catch2 by default.
option(OBJECT_PLANNER_BUILD_TESTS "Build the C++ unit tests" OFF)
if(OBJECT_PLANNER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
