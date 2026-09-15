cmake_minimum_required(VERSION 3.16)
project(leg_to_wheel_checks LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# RTNeural declares an older minimum; retain its compatibility on CMake 4.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(PKG "${REPO}/ros2_ws/src/neural_controller")
set(RTNEURAL_EIGEN ON CACHE BOOL "Match robot backend" FORCE)
add_subdirectory("${PKG}/modules/RTNeural" "${CMAKE_CURRENT_BINARY_DIR}/rtneural")
add_executable(leg_to_wheel_test "${PKG}/test/leg_to_wheel_policy_test.cpp")
target_include_directories(leg_to_wheel_test PRIVATE "${PKG}/include")
target_link_libraries(leg_to_wheel_test PRIVATE RTNeural)
enable_testing()
add_test(NAME leg_to_wheel_export_and_runtime COMMAND leg_to_wheel_test
  "${PKG}/launch/policy_leg_to_wheel.json" "${PKG}/test/leg_to_wheel_policy_reference.csv"
  "${REPO}/policies/leg_to_wheel/evidence/runtime_reference.csv")
