# C++ unit tests for object_planner. Off by default; enable with
#   cmake -DOBJECT_PLANNER_BUILD_TESTS=ON -B build && cmake --build build
#   ctest --test-dir build
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(Catch2)

add_executable(object_planner_tests
  test_sphere_tree_builder.cpp
  test_batched_collision_checker.cpp
  test_rrt_connect.cpp
  test_bit_star.cpp
  test_simplify.cpp
)
target_link_libraries(object_planner_tests PRIVATE
  object_planner::object_planner
  Catch2::Catch2WithMain
)
target_compile_features(object_planner_tests PRIVATE cxx_std_17)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(object_planner_tests)
