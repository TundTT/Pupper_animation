cmake_minimum_required(VERSION 3.8)
project(control_board_hardware_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(robot_calibration REQUIRED)
find_package(pluginlib REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2 REQUIRED)

# add compile directive for shared library
add_library(control_board_hardware_interface SHARED
  src/control_board_hardware_interface.cpp
  src/rt/rt_spi.cpp
  src/rt/rt_bno055.cpp
  src/imu_manager.cpp
)

target_compile_features(control_board_hardware_interface PUBLIC cxx_std_17)

# add include directory
target_include_directories(control_board_hardware_interface
  PRIVATE
    include
)

ament_target_dependencies(
  control_board_hardware_interface
  robot_calibration
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  tf2
)

pluginlib_export_plugin_description_file(hardware_interface control_board_hardware_interface.xml)

# INSTALL
install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
)
install(
  DIRECTORY include/
  DESTINATION include
)
## EXPORTS
ament_export_include_directories(
  include
)
ament_export_libraries(
  ${PROJECT_NAME}
)
ament_export_dependencies(
  robot_calibration
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  transmission_interface
)
install(FILES runtime_bundle.txt DESTINATION share/${PROJECT_NAME})
if(BUILD_TESTING)
  enable_testing()
  add_executable(spi_feedback_test test/spi_feedback_test.cpp src/rt/rt_spi.cpp)
  target_compile_features(spi_feedback_test PRIVATE cxx_std_17)
  target_include_directories(spi_feedback_test PRIVATE include)
  target_link_options(spi_feedback_test PRIVATE -Wl,--wrap=ioctl)
  target_link_libraries(spi_feedback_test pthread)
  add_test(NAME spi_feedback_validation COMMAND spi_feedback_test)
  add_executable(gravity_calibration_test test/gravity_calibration_test.cpp)
  target_compile_features(gravity_calibration_test PRIVATE cxx_std_17)
  target_include_directories(gravity_calibration_test PRIVATE include)
  add_test(NAME gravity_calibration COMMAND gravity_calibration_test)
  add_executable(gravity_hardware_test test/gravity_hardware_test.cpp
    src/control_board_hardware_interface.cpp src/imu_manager.cpp src/rt/rt_bno055.cpp)
  target_compile_features(gravity_hardware_test PRIVATE cxx_std_17)
  target_include_directories(gravity_hardware_test PRIVATE include)
  ament_target_dependencies(gravity_hardware_test robot_calibration hardware_interface pluginlib rclcpp rclcpp_lifecycle tf2)
  add_test(NAME gravity_hardware_fake_transport COMMAND gravity_hardware_test)
  set_tests_properties(gravity_hardware_fake_transport PROPERTIES TIMEOUT 25 ENVIRONMENT "ROS_DOMAIN_ID=218;ROS_LOCALHOST_ONLY=1")
endif()
ament_package()