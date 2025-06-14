add_library(object_planner_core
    sphere_tree_builder.cpp
    batched_collision_checker.cpp
    rrt_star.cpp
    path_smoother.cpp
)

set_target_properties(object_planner_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Use target_include_directories to specify search paths for this target.
# PUBLIC: Consumers of this library also get these include directories.
target_include_directories(object_planner_core PUBLIC
    # Add the current directory (src) so we can do #include "data_structures.h"
    ${CMAKE_CURRENT_SOURCE_DIR} 
    
    # Add the root of the nanoflann headers
    ${CMAKE_SOURCE_DIR}/extern/nanoflann/include 
)

# Link to other libraries if needed. Eigen is header-only, so linking isn't
# strictly necessary but adding the include dir is.
target_include_directories(object_planner_core PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${xsimd_SOURCE_DIR}/include
)