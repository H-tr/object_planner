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

# --- Python bindings ------------------------------------------------------
# nanobind: locate via its CMake helper. scikit-build-core ensures the
# matching Python interpreter is set up before this point.
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
install(DIRECTORY include/ DESTINATION include)
