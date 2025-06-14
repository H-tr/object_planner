cmake_minimum_required(VERSION 3.15)
project(ObjectPlanner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)

# --- Fetch xsimd (SIMD library) ---
include(FetchContent)
FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 8.1.0
)
FetchContent_MakeAvailable(xsimd)

# Add our C++ source library
add_subdirectory(src)

# Define the Python module from the bindings file
pybind11_add_module(object_planner_py SHARED py_src/object_planner_py/py_bindings.cpp)

# Because object_planner_core has PUBLIC include directories, linking to it
# is all we need to do. The include paths for Eigen, xsimd, nanoflann, and src
# will be inherited automatically.
target_link_libraries(object_planner_py PRIVATE object_planner_core)

# Install the compiled module into the python package directory
install(TARGETS object_planner_py DESTINATION object_planner_py)