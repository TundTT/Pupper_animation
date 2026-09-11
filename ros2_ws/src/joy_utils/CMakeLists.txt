cmake_minimum_required(VERSION 3.5)
project(joy_utils)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(robot_calibration REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)

# Add the executable
add_executable(estop_controller src/estop_controller.cpp)

# Link against ROS 2 libraries
ament_target_dependencies(estop_controller
  robot_calibration
  rclcpp
  sensor_msgs
  std_msgs
  controller_manager_msgs
)

# Install the executable
install(TARGETS estop_controller
  DESTINATION lib/${PROJECT_NAME}
)

# Export package
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(alignment_joystick test/alignment_joystick_test.py
    ENV "ALIGN_JOY_TEST_EXE=$<TARGET_FILE:estop_controller>" TIMEOUT 45)
endif()

ament_package()
