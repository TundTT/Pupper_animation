cmake_minimum_required(VERSION 3.8)
project(neural_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(neural_controller SHARED src/neural_controller.cpp src/keyframe_controller.cpp src/inverted_triangle_controller.cpp src/triangle_roll_controller.cpp)
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

add_library(joint_pose_controller SHARED src/joint_pose_controller.cpp)
target_compile_features(joint_pose_controller PUBLIC cxx_std_17)
target_include_directories(joint_pose_controller PRIVATE include)
ament_target_dependencies(joint_pose_controller controller_interface hardware_interface pluginlib robot_calibration rclcpp rclcpp_lifecycle sensor_msgs std_msgs)
install(TARGETS joint_pose_controller DESTINATION lib)

# Separate library: adding this diagnostic never requires overwriting the
# existing neural_controller library used by a live hardware session.
add_library(hub_roll_controller SHARED src/hub_roll_controller.cpp)
target_compile_features(hub_roll_controller PUBLIC cxx_std_17)
target_include_directories(hub_roll_controller PRIVATE include)
target_link_libraries(hub_roll_controller neural_controller neural_controller_parameters RTNeural)
ament_target_dependencies(hub_roll_controller controller_interface robot_calibration hardware_interface pluginlib realtime_tools rclcpp rclcpp_lifecycle geometry_msgs sensor_msgs tf2)
install(TARGETS hub_roll_controller DESTINATION lib)

# Candidate plugin is separate; existing policy loaders and defaults are unchanged.
add_library(notebook_lift_controller SHARED src/notebook_lift_controller.cpp)
target_compile_features(notebook_lift_controller PUBLIC cxx_std_17)
target_include_directories(notebook_lift_controller PRIVATE include)
target_link_libraries(notebook_lift_controller neural_controller neural_controller_parameters RTNeural)
ament_target_dependencies(notebook_lift_controller controller_interface robot_calibration hardware_interface pluginlib realtime_tools rclcpp rclcpp_lifecycle geometry_msgs sensor_msgs tf2 std_msgs)
install(TARGETS notebook_lift_controller DESTINATION lib)


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
  add_executable(leg_to_wheel_policy_test test/leg_to_wheel_policy_test.cpp)
  target_compile_features(leg_to_wheel_policy_test PRIVATE cxx_std_17)
  target_include_directories(leg_to_wheel_policy_test PRIVATE include)
  target_link_libraries(leg_to_wheel_policy_test RTNeural)
  add_test(NAME leg_to_wheel_policy_and_runtime COMMAND leg_to_wheel_policy_test
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_leg_to_wheel.json
    ${CMAKE_CURRENT_SOURCE_DIR}/test/leg_to_wheel_policy_reference.csv
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../policies/leg_to_wheel/evidence/runtime_reference.csv)
  add_executable(notebook_lift_align_test test/notebook_lift_align_test.cpp)
  target_compile_features(notebook_lift_align_test PRIVATE cxx_std_17)
  target_include_directories(notebook_lift_align_test PRIVATE include)
  add_test(NAME notebook_lift_align COMMAND notebook_lift_align_test ${CMAKE_CURRENT_SOURCE_DIR}/../../../hardware_testing/notebook_lift/align_geometry_reference.csv)
  foreach(kind IN ITEMS observation targets)
    add_executable(notebook_lift_${kind}_test test/notebook_lift_${kind}_test.cpp)
    target_compile_features(notebook_lift_${kind}_test PRIVATE cxx_std_17)
    target_include_directories(notebook_lift_${kind}_test PRIVATE include)
  endforeach()
  add_test(NAME notebook_lift_observation COMMAND notebook_lift_observation_test ${CMAKE_CURRENT_SOURCE_DIR}/../../../hardware_testing/notebook_lift/observation_reference.csv)
  add_test(NAME notebook_lift_targets COMMAND notebook_lift_targets_test ${CMAKE_CURRENT_SOURCE_DIR}/../../../hardware_testing/notebook_lift/target_reference.csv)
  add_executable(notebook_lift_core_test test/notebook_lift_core_test.cpp)
  target_compile_features(notebook_lift_core_test PRIVATE cxx_std_17)
  target_include_directories(notebook_lift_core_test PRIVATE include)
  target_link_libraries(notebook_lift_core_test RTNeural)
  add_test(NAME notebook_lift_core COMMAND notebook_lift_core_test
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_notebook_lift.json
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../hardware_testing/notebook_lift/network_reference.csv
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../hardware_testing/notebook_lift/geometry_reference.csv)
  add_executable(notebook_lift_controller_test test/notebook_lift_controller_test.cpp)
  target_compile_features(notebook_lift_controller_test PRIVATE cxx_std_17)
  target_include_directories(notebook_lift_controller_test PRIVATE include)
  target_link_libraries(notebook_lift_controller_test notebook_lift_controller neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(notebook_lift_controller_test controller_interface hardware_interface robot_calibration rclcpp rclcpp_lifecycle realtime_tools geometry_msgs tf2 sensor_msgs std_msgs)
  add_test(NAME notebook_lift_lifecycle COMMAND notebook_lift_controller_test
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/notebook_lift_config.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_notebook_lift.json)
  add_test(NAME notebook_lift_align_lifecycle COMMAND notebook_lift_controller_test
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/notebook_lift_align_config.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_notebook_lift.json)
  set_tests_properties(notebook_lift_lifecycle notebook_lift_align_lifecycle PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=194;ROS_LOCALHOST_ONLY=1")
  add_executable(walking_frame_test test/walking_frame_test.cpp)
  target_compile_features(walking_frame_test PRIVATE cxx_std_17)
  target_include_directories(walking_frame_test PRIVATE include)
  add_test(NAME walking_frame COMMAND walking_frame_test)
  add_executable(walking_handoff_test test/walking_handoff_test.cpp)
  target_compile_features(walking_handoff_test PRIVATE cxx_std_17)
  target_include_directories(walking_handoff_test PRIVATE include)
  target_link_libraries(walking_handoff_test neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(walking_handoff_test controller_interface hardware_interface robot_calibration rclcpp rclcpp_lifecycle realtime_tools geometry_msgs sensor_msgs tf2)
  add_test(NAME walking_handoff COMMAND walking_handoff_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_walk_v2.json)
  set_tests_properties(walking_handoff PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=193;ROS_LOCALHOST_ONLY=1")
  add_executable(joint_pose_test test/joint_pose_test.cpp)
  target_compile_features(joint_pose_test PRIVATE cxx_std_17)
  target_include_directories(joint_pose_test PRIVATE include)
  add_test(NAME joint_pose_recovery COMMAND joint_pose_test)
  add_executable(joint_pose_controller_test test/joint_pose_controller_test.cpp)
  target_compile_features(joint_pose_controller_test PRIVATE cxx_std_17)
  target_include_directories(joint_pose_controller_test PRIVATE include)
  ament_target_dependencies(joint_pose_controller_test controller_interface hardware_interface rclcpp rclcpp_lifecycle pluginlib sensor_msgs std_msgs robot_calibration)
  add_test(NAME joint_pose_lifecycle COMMAND joint_pose_controller_test)
  set_tests_properties(joint_pose_lifecycle PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=193;ROS_LOCALHOST_ONLY=1")
  add_executable(measured_roll_test test/measured_roll_test.cpp)
  target_compile_features(measured_roll_test PRIVATE cxx_std_17)
  target_include_directories(measured_roll_test PRIVATE include)
  target_link_libraries(measured_roll_test RTNeural)
  add_test(NAME measured_roll_core COMMAND measured_roll_test)
  add_executable(hub_roll_controller_test test/hub_roll_controller_test.cpp)
  target_compile_features(hub_roll_controller_test PRIVATE cxx_std_17)
  target_include_directories(hub_roll_controller_test PRIVATE include)
  target_link_libraries(hub_roll_controller_test hub_roll_controller neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(hub_roll_controller_test controller_interface hardware_interface rclcpp rclcpp_lifecycle robot_calibration realtime_tools geometry_msgs tf2 sensor_msgs)
  add_test(NAME hub_roll_lifecycle COMMAND hub_roll_controller_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/hub_roll_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_plan.json)
  add_executable(triangle_roll_test test/triangle_roll_test.cpp)
  target_compile_features(triangle_roll_test PRIVATE cxx_std_17)
  target_include_directories(triangle_roll_test PRIVATE include)
  target_link_libraries(triangle_roll_test RTNeural)
  add_test(NAME triangle_roll_core COMMAND triangle_roll_test ${CMAKE_CURRENT_SOURCE_DIR}/test/triangle_roll_support_reference.csv)
  add_executable(triangle_roll_controller_test test/triangle_roll_controller_test.cpp)
  target_compile_features(triangle_roll_controller_test PRIVATE cxx_std_17)
  target_include_directories(triangle_roll_controller_test PRIVATE include)
  target_link_libraries(triangle_roll_controller_test neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(triangle_roll_controller_test controller_interface hardware_interface rclcpp rclcpp_lifecycle robot_calibration realtime_tools geometry_msgs tf2 sensor_msgs)
  add_test(NAME triangle_roll_lifecycle COMMAND triangle_roll_controller_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_plan.json)
  add_test(NAME triangle_roll_stand_lifecycle COMMAND triangle_roll_controller_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_plan.json ${CMAKE_CURRENT_SOURCE_DIR}/launch/triangle_roll_stand.yaml)

  add_executable(inverted_triangle_test test/inverted_triangle_test.cpp)
  target_compile_features(inverted_triangle_test PRIVATE cxx_std_17)
  target_include_directories(inverted_triangle_test PRIVATE include)
  target_link_libraries(inverted_triangle_test RTNeural)
  add_test(NAME inverted_triangle_core COMMAND inverted_triangle_test ${CMAKE_CURRENT_SOURCE_DIR}/launch/inverted_triangle_plan.json)
  add_executable(inverted_triangle_controller_test test/inverted_triangle_controller_test.cpp)
  target_compile_features(inverted_triangle_controller_test PRIVATE cxx_std_17)
  target_include_directories(inverted_triangle_controller_test PRIVATE include)
  target_link_libraries(inverted_triangle_controller_test neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(inverted_triangle_controller_test controller_interface hardware_interface rclcpp rclcpp_lifecycle robot_calibration realtime_tools geometry_msgs tf2 sensor_msgs)
  add_test(NAME inverted_triangle_lifecycle COMMAND inverted_triangle_controller_test
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/inverted_triangle_config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/inverted_triangle_plan.json)
  add_executable(keyframe_position_test test/keyframe_position_test.cpp)
  target_compile_features(keyframe_position_test PRIVATE cxx_std_17)
  target_include_directories(keyframe_position_test PRIVATE include)
  add_test(NAME keyframe_position COMMAND keyframe_position_test)
  add_executable(keyframe_controller_test test/keyframe_controller_test.cpp)
  target_compile_features(keyframe_controller_test PRIVATE cxx_std_17)
  target_include_directories(keyframe_controller_test PRIVATE include)
  target_link_libraries(keyframe_controller_test neural_controller neural_controller_parameters RTNeural)
  ament_target_dependencies(keyframe_controller_test controller_interface hardware_interface rclcpp rclcpp_lifecycle robot_calibration realtime_tools geometry_msgs tf2)
  add_test(NAME keyframe_controller_lifecycle COMMAND keyframe_controller_test
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/launch/keyframe_config.json)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(startup_launch test/startup_launch_test.py)
  ament_add_pytest_test(locomotion_manager test/locomotion_manager_test.py TIMEOUT 90
    ENV "ROS_DOMAIN_ID=95")
  ament_add_pytest_test(keyframe_manager test/keyframe_manager_test.py TIMEOUT 90
    ENV ROS_DOMAIN_ID=181 ROS_LOCALHOST_ONLY=1)
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
  add_executable(wheel_policy_test test/wheel_policy_test.cpp)
  target_compile_features(wheel_policy_test PRIVATE cxx_std_17)
  target_link_libraries(wheel_policy_test RTNeural)
  add_test(NAME wheel_policy_contract_and_inference
    COMMAND wheel_policy_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_wheel.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/wheel_policy_reference.csv)
  add_executable(walk_policy_test test/walk_policy_test.cpp)
  target_compile_features(walk_policy_test PRIVATE cxx_std_17)
  target_include_directories(walk_policy_test PRIVATE include)
  target_link_libraries(walk_policy_test RTNeural)
  add_test(NAME walk_policy_contract_and_inference
    COMMAND walk_policy_test
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/policy_walk_v2.json
      ${CMAKE_CURRENT_SOURCE_DIR}/test/walk_policy_reference.csv)
  option(QUADMORPH_FOCUSED_TESTS "Configure controller tests without recursive vendor lint discovery" OFF)
  if(NOT QUADMORPH_FOCUSED_TESTS)
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
endif()

ament_package()
