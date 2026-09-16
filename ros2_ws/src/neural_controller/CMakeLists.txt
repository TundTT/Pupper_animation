cmake_minimum_required(VERSION 3.8)
project(neural_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(neural_controller SHARED src/neural_controller.cpp src/keyframe_controller.cpp src/triangle_roll_controller.cpp)
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
find_package(std_msgs REQUIRED)
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
  sensor_msgs
  tf2
)

pluginlib_export_plugin_description_file(controller_interface neural_controller.xml)

add_library(wheel_lift_controller SHARED src/wheel_lift_controller.cpp)
target_compile_features(wheel_lift_controller PUBLIC cxx_std_17)
target_include_directories(wheel_lift_controller PRIVATE include)
target_link_libraries(wheel_lift_controller neural_controller neural_controller_parameters RTNeural)
ament_target_dependencies(wheel_lift_controller controller_interface robot_calibration hardware_interface pluginlib realtime_tools rclcpp rclcpp_lifecycle geometry_msgs sensor_msgs tf2 std_msgs)
install(TARGETS wheel_lift_controller DESTINATION lib)

add_library(leg_to_wheel_controller SHARED src/leg_to_wheel_controller.cpp)
target_compile_features(leg_to_wheel_controller PUBLIC cxx_std_17)
target_include_directories(leg_to_wheel_controller PRIVATE include)
target_link_libraries(leg_to_wheel_controller neural_controller neural_controller_parameters RTNeural)
ament_target_dependencies(leg_to_wheel_controller controller_interface robot_calibration hardware_interface pluginlib realtime_tools rclcpp rclcpp_lifecycle geometry_msgs sensor_msgs tf2 std_msgs)
install(TARGETS leg_to_wheel_controller DESTINATION lib)

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
install(PROGRAMS scripts/motion_buttons.py DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  enable_testing()
  add_executable(walk_policy_test test/walk_policy_test.cpp)
  target_compile_features(walk_policy_test PRIVATE cxx_std_17)
  target_include_directories(walk_policy_test PRIVATE include)
  target_link_libraries(walk_policy_test RTNeural)
  add_test(NAME walk_policy_test COMMAND walk_policy_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_walk_v2.json ${CMAKE_CURRENT_SOURCE_DIR}/test/walk_policy_reference.csv)
  add_executable(wheel_policy_test test/wheel_policy_test.cpp)
  target_compile_features(wheel_policy_test PRIVATE cxx_std_17)
  target_include_directories(wheel_policy_test PRIVATE include)
  target_link_libraries(wheel_policy_test RTNeural)
  add_test(NAME wheel_policy_test COMMAND wheel_policy_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_wheel.json ${CMAKE_CURRENT_SOURCE_DIR}/test/wheel_policy_reference.csv)
  add_executable(leg_to_wheel_policy_test test/leg_to_wheel_policy_test.cpp)
  target_compile_features(leg_to_wheel_policy_test PRIVATE cxx_std_17)
  target_include_directories(leg_to_wheel_policy_test PRIVATE include)
  target_link_libraries(leg_to_wheel_policy_test RTNeural)
  add_test(NAME leg_to_wheel_policy_test COMMAND leg_to_wheel_policy_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_leg_to_wheel.json ${CMAKE_CURRENT_SOURCE_DIR}/test/leg_to_wheel_policy_reference.csv ${CMAKE_CURRENT_SOURCE_DIR}/../../../policies/leg_to_wheel/evidence/runtime_reference.csv)
  add_executable(triangle_roll_test test/triangle_roll_test.cpp)
  target_compile_features(triangle_roll_test PRIVATE cxx_std_17)
  target_include_directories(triangle_roll_test PRIVATE include)
  target_link_libraries(triangle_roll_test RTNeural)
  add_test(NAME triangle_roll_test COMMAND triangle_roll_test ${CMAKE_CURRENT_SOURCE_DIR}/test/triangle_roll_support_reference.csv)
  add_executable(triangle_roll_controller_test test/triangle_roll_controller_test.cpp)
  target_compile_features(triangle_roll_controller_test PRIVATE cxx_std_17)
  target_include_directories(triangle_roll_controller_test PRIVATE include)
  target_link_libraries(triangle_roll_controller_test RTNeural)
  target_link_libraries(triangle_roll_controller_test neural_controller wheel_lift_controller neural_controller_parameters)
  ament_target_dependencies(triangle_roll_controller_test controller_interface hardware_interface robot_calibration rclcpp rclcpp_lifecycle realtime_tools geometry_msgs tf2 sensor_msgs std_msgs)
  add_test(NAME triangle_roll_controller_test COMMAND triangle_roll_controller_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_plan.json)
  add_executable(walking_frame_test test/walking_frame_test.cpp)
  target_compile_features(walking_frame_test PRIVATE cxx_std_17)
  target_include_directories(walking_frame_test PRIVATE include)
  target_link_libraries(walking_frame_test RTNeural)
  add_test(NAME walking_frame_test COMMAND walking_frame_test )
  add_executable(walking_handoff_test test/walking_handoff_test.cpp)
  target_compile_features(walking_handoff_test PRIVATE cxx_std_17)
  target_include_directories(walking_handoff_test PRIVATE include)
  target_link_libraries(walking_handoff_test RTNeural)
  target_link_libraries(walking_handoff_test neural_controller wheel_lift_controller neural_controller_parameters)
  ament_target_dependencies(walking_handoff_test controller_interface hardware_interface robot_calibration rclcpp rclcpp_lifecycle realtime_tools geometry_msgs tf2 sensor_msgs std_msgs)
  add_test(NAME walking_handoff_test COMMAND walking_handoff_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_walk_v2.json)
  add_executable(wheel_lift_core_test test/wheel_lift_core_test.cpp)
  target_compile_features(wheel_lift_core_test PRIVATE cxx_std_17)
  target_include_directories(wheel_lift_core_test PRIVATE include)
  target_link_libraries(wheel_lift_core_test RTNeural)
  add_test(NAME wheel_lift_core_test COMMAND wheel_lift_core_test ${CMAKE_CURRENT_SOURCE_DIR}/../../../policies/leg_lift_wheel/geometry_reference.csv)
  add_executable(wheel_lift_controller_test test/wheel_lift_controller_test.cpp)
  target_compile_features(wheel_lift_controller_test PRIVATE cxx_std_17)
  target_include_directories(wheel_lift_controller_test PRIVATE include)
  target_link_libraries(wheel_lift_controller_test RTNeural)
  target_link_libraries(wheel_lift_controller_test neural_controller wheel_lift_controller neural_controller_parameters)
  ament_target_dependencies(wheel_lift_controller_test controller_interface hardware_interface robot_calibration rclcpp rclcpp_lifecycle realtime_tools geometry_msgs tf2 sensor_msgs std_msgs)
  add_test(NAME wheel_lift_controller_test COMMAND wheel_lift_controller_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/wheel_lift_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_leg_lift_wheel.json)
  add_executable(leg_to_wheel_controller_test test/leg_to_wheel_controller_test.cpp)
  target_compile_features(leg_to_wheel_controller_test PRIVATE cxx_std_17)
  target_include_directories(leg_to_wheel_controller_test PRIVATE include)
  target_link_libraries(leg_to_wheel_controller_test neural_controller leg_to_wheel_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(leg_to_wheel_controller_test controller_interface hardware_interface robot_calibration rclcpp rclcpp_lifecycle realtime_tools geometry_msgs tf2 sensor_msgs std_msgs)
  add_test(NAME leg_to_wheel_controller_test COMMAND leg_to_wheel_controller_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/leg_to_wheel_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_leg_to_wheel.json)
  add_executable(leg_to_wheel_manual_test test/leg_to_wheel_manual_test.cpp)
  target_compile_features(leg_to_wheel_manual_test PRIVATE cxx_std_17)
  target_include_directories(leg_to_wheel_manual_test PRIVATE include)
  target_link_libraries(leg_to_wheel_manual_test RTNeural)
  add_test(NAME leg_to_wheel_manual_test COMMAND leg_to_wheel_manual_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_leg_to_wheel.json ${CMAKE_CURRENT_SOURCE_DIR}/../../../policies/leg_to_wheel/evidence/runtime_reference.csv)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(motion_buttons test/test_motion_buttons.py)
  ament_add_pytest_test(gravity_runtime_description test/gravity_runtime_test.py)
  ament_add_pytest_test(combined_runtime_mock test/combined_wheel_lift_manager_test.py TIMEOUT 90
    ENV ROS_DOMAIN_ID=217 ROS_LOCALHOST_ONLY=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1)
endif()

install(FILES runtime_bundle.txt DESTINATION share/${PROJECT_NAME})
ament_package()
