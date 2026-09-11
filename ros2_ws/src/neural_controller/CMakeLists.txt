cmake_minimum_required(VERSION 3.8)
project(neural_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(neural_controller SHARED src/neural_controller.cpp)
target_compile_features(neural_controller PUBLIC cxx_std_17)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(robot_calibration REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(generate_parameter_library REQUIRED)

generate_parameter_library(
  neural_controller_parameters # cmake target name for the parameter library
  src/neural_controller_parameters.yaml # path to input yaml file
)

set(RTNEURAL_EIGEN ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(modules/RTNeural)

target_link_libraries(neural_controller neural_controller_parameters RTNeural)

target_include_directories(neural_controller PRIVATE include)

ament_target_dependencies(
  neural_controller
  controller_interface
  robot_calibration
  hardware_interface
  pluginlib
  realtime_tools
  rclcpp
  rclcpp_lifecycle
  geometry_msgs
  tf2
)

pluginlib_export_plugin_description_file(controller_interface neural_controller.xml)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(startup_launch test/startup_launch_test.py)
  add_executable(align_motion_test test/align_motion_test.cpp)
  target_compile_features(align_motion_test PRIVATE cxx_std_17)
  target_include_directories(align_motion_test PRIVATE include)
  add_test(NAME align_v5_motion_contract COMMAND align_motion_test)
  add_executable(align_export_test test/align_export_test.cpp)
  target_compile_features(align_export_test PRIVATE cxx_std_17)
  target_link_libraries(align_export_test RTNeural)
  add_test(NAME align_v5_trained_export
    COMMAND align_export_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_wheel_align_motion_v5.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/align_v5_reference.csv)
  add_executable(hybrid_controller_test test/hybrid_controller_test.cpp)
  target_compile_features(hybrid_controller_test PRIVATE cxx_std_17)
  target_include_directories(hybrid_controller_test PRIVATE include)
  target_link_libraries(hybrid_controller_test neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(hybrid_controller_test controller_interface hardware_interface
    realtime_tools rclcpp rclcpp_lifecycle geometry_msgs tf2 robot_calibration)
  add_test(NAME hybrid_controller_lifecycle
    COMMAND hybrid_controller_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/config.yaml
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_wheel_align_hybrid.json)
  add_test(NAME align_v5_controller_lifecycle
    COMMAND hybrid_controller_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/config.yaml
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_wheel_align_motion_v5.json)
  add_executable(hybrid_policy_test test/hybrid_policy_test.cpp)
  target_compile_features(hybrid_policy_test PRIVATE cxx_std_17)
  target_include_directories(hybrid_policy_test PRIVATE include)
  target_link_libraries(hybrid_policy_test RTNeural)
  add_test(NAME hybrid_policy_contract_and_inference
    COMMAND hybrid_policy_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_wheel_align_hybrid.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/hybrid_policy_reference.csv)
  add_executable(walk_policy_test test/walk_policy_test.cpp)
  target_compile_features(walk_policy_test PRIVATE cxx_std_17)
  target_include_directories(walk_policy_test PRIVATE include)
  target_link_libraries(walk_policy_test RTNeural)
  add_test(NAME walk_policy_contract_and_inference
    COMMAND walk_policy_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_walk_v2.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/walk_policy_reference.csv)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
