#include "control_board_hardware_interface/control_board_hardware_interface.hpp"

#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace control_board_hardware_interface {

const char *RED_ANSI = "\033[1;31m";
const char *YELLOW_ANSI = "\033[1;33m";
const char *RESET_ANSI = "\033[0m";

ControlBoardHardwareInterface::~ControlBoardHardwareInterface() {
  // Release lock if we have it
  if (lock_fd_ != -1) {
    flock(lock_fd_, LOCK_UN);
    close(lock_fd_);
    unlink("/tmp/control_board_hardware.lock");  // Delete the lock file
  }

  // Deactivate everything when ctrl-c is pressed
  on_deactivate(rclcpp_lifecycle::State());
}

hardware_interface::CallbackReturn ControlBoardHardwareInterface::on_init(
    const hardware_interface::HardwareInfo &info) {
  lock_fd_ = open("/tmp/control_board_hardware.lock", O_CREAT, 0666);
  if (lock_fd_ == -1) {
    RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"), "Failed to open lock file");
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (flock(lock_fd_, LOCK_EX | LOCK_NB) == -1) {
    RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
                 "Another instance is already running");
    close(lock_fd_);
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (hardware_interface::SystemInterface::on_init(info) !=
      hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  hw_state_positions_.resize(info_.joints.size(), 0.0);
  hw_state_velocities_.resize(info_.joints.size(), 0.0);
  hw_state_efforts_.resize(info_.joints.size(), 0.0);

  hw_command_positions_.resize(info_.joints.size(), 0.0);
  hw_command_velocities_.resize(info_.joints.size(), 0.0);
  hw_command_efforts_.resize(info_.joints.size(), 0.0);
  hw_command_kps_.resize(info_.joints.size(), 0.0);
  hw_command_kds_.resize(info_.joints.size(), 0.0);

  for (const hardware_interface::ComponentInfo &joint : info_.joints) {
    // Set params for each joint
    hw_actuator_can_channels_.push_back(std::stoi(joint.parameters.at("can_channel")));
    hw_actuator_can_ids_.push_back(std::stoi(joint.parameters.at("can_id")));

    // Set limits for each joint
    hw_actuator_position_mins_.push_back(std::stod(joint.parameters.at("position_min")));
    hw_actuator_position_maxs_.push_back(std::stod(joint.parameters.at("position_max")));
    hw_actuator_velocity_maxs_.push_back(std::stod(joint.parameters.at("velocity_max")));

    // hard_limit_min/max is OPTIONAL -- most joints don't declare it and get +-infinity
    // (the unconditional end-stop in copy_actuator_commands() never triggers). Only
    // joints with a real "this breaks the hardware if driven too far" constraint should
    // set a finite value. Deliberately separate from position_min/max above -- see the
    // member comment in the header for why reusing that would be wrong.
    if (joint.parameters.count("hard_limit_min")) {
      hw_actuator_hard_limit_mins_.push_back(std::stod(joint.parameters.at("hard_limit_min")));
    } else {
      hw_actuator_hard_limit_mins_.push_back(-std::numeric_limits<double>::infinity());
    }
    if (joint.parameters.count("hard_limit_max")) {
      hw_actuator_hard_limit_maxs_.push_back(std::stod(joint.parameters.at("hard_limit_max")));
    } else {
      hw_actuator_hard_limit_maxs_.push_back(std::numeric_limits<double>::infinity());
    }
    hw_actuator_hard_limit_active_max_.push_back(false);
    hw_actuator_hard_limit_active_min_.push_back(false);
    hw_actuator_predicting_.push_back(false);
    hw_actuator_effort_maxs_.push_back(std::stod(joint.parameters.at("effort_max")));
    hw_actuator_kp_maxs_.push_back(std::stod(joint.parameters.at("kp_max")));
    hw_actuator_kd_maxs_.push_back(std::stod(joint.parameters.at("kd_max")));

    // Homing parameters
    hw_actuator_homing_stages_.push_back(std::stoi(joint.parameters.at("homing_stage")));
    hw_actuator_homing_velocities_.push_back(std::stod(joint.parameters.at("homing_velocity")));
    hw_actuator_homing_kps_.push_back(std::stod(joint.parameters.at("homing_kp")));
    hw_actuator_homing_kds_.push_back(std::stod(joint.parameters.at("homing_kd")));
    hw_actuator_homed_positions_.push_back(std::stod(joint.parameters.at("homed_position")));
    hw_actuator_post_homing_positions_.push_back(
        std::stod(joint.parameters.at("post_homing_position")));
    hw_actuator_zero_positions_.push_back(0.0);
    hw_actuator_homing_torque_thresholds_.push_back(
        std::stod(joint.parameters.at("homing_torque_threshold")));
    hw_actuator_is_homed_.push_back(false);

    // homing_reference_raw is OPTIONAL -- a hardware-measured raw sensor reading (taken
    // before the homed_position snap) from a boot where the joint was confirmed to be
    // resting in the correct gravity-drop pose. Only meaningful for disarmed-homing
    // joints (homing_torque_threshold == 0), where nothing else validates that the joint
    // was actually in that pose at boot. Absent (NaN) disables the check for this joint.
    if (joint.parameters.count("homing_reference_raw")) {
      hw_actuator_homing_reference_raw_.push_back(
          std::stod(joint.parameters.at("homing_reference_raw")));
    } else {
      hw_actuator_homing_reference_raw_.push_back(std::numeric_limits<double>::quiet_NaN());
    }
  }

  if (info_.hardware_parameters.count("use_imu") &&
      (info_.hardware_parameters.at("use_imu") == "false" ||
       info_.hardware_parameters.at("use_imu") == "False")) {
    RCLCPP_WARN(rclcpp::get_logger("ControlBoardHardwareInterface"), "%sIMU not enabled%s",
                YELLOW_ANSI, RESET_ANSI);
    imu_manager_.enabled = false;
  } else {
    imu_manager_.enabled = true;

    // Set up IMU parameters
    double imu_roll = std::stod(info_.sensors[0].parameters.at("roll"));
    double imu_pitch = std::stod(info_.sensors[0].parameters.at("pitch"));
    double imu_yaw = std::stod(info_.sensors[0].parameters.at("yaw"));
    imu_manager_.set_imu_offset(imu_roll, imu_pitch, imu_yaw);

    // Set up the IMU
    // TODO: make micros_between_reports a parameter
    imu_manager_.begin(/*micros_between_reports=*/10000);
  }

  // Set up SPI
  init_spi();
  spi_command_ = get_spi_command();
  spi_data_ = get_spi_data();

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
ControlBoardHardwareInterface::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Add joint state interfaces
  for (auto i = 0u; i < info_.joints.size(); i++) {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_state_positions_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_state_velocities_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_state_efforts_[i]));
  }

  // Add IMU state interfaces
  state_interfaces.emplace_back(hardware_interface::StateInterface("imu_sensor", "orientation.x",
                                                                   &hw_state_imu_orientation_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface("imu_sensor", "orientation.y",
                                                                   &hw_state_imu_orientation_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface("imu_sensor", "orientation.z",
                                                                   &hw_state_imu_orientation_[2]));
  state_interfaces.emplace_back(hardware_interface::StateInterface("imu_sensor", "orientation.w",
                                                                   &hw_state_imu_orientation_[3]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "angular_velocity.x", &hw_state_imu_angular_velocity_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "angular_velocity.y", &hw_state_imu_angular_velocity_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "angular_velocity.z", &hw_state_imu_angular_velocity_[2]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "linear_acceleration.x", &hw_state_imu_linear_acceleration_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "linear_acceleration.y", &hw_state_imu_linear_acceleration_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "linear_acceleration.z", &hw_state_imu_linear_acceleration_[2]));

  // TODO: Add once int64 type is supported
  //   state_interfaces.emplace_back(
  //       hardware_interface::StateInterface("imu_sensor", "imu_packet_timestamp",
  //       &imu_packet_timestamp_micros_));
  //   state_interfaces.emplace_back(hardware_interface::StateInterface(
  //       "imu_sensor", "imu_measurement_timestamp", &imu_measurement_timestamp_micros_));
  state_interfaces.emplace_back(hardware_interface::StateInterface(
      "imu_sensor", "time_since_measurement_seconds", &imu_time_since_measurement_seconds_));

  return state_interfaces;
}

hardware_interface::CallbackReturn ControlBoardHardwareInterface::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  // reset values always when configuring hardware
  for (uint i = 0; i < hw_state_positions_.size(); i++) {
    hw_state_positions_[i] = 0.0;
    hw_state_velocities_[i] = 0.0;
    hw_state_efforts_[i] = 0.0;
    hw_command_positions_[i] = 0.0;
    hw_command_velocities_[i] = 0.0;
    hw_command_efforts_[i] = 0.0;
    hw_command_kps_[i] = 0.0;
    hw_command_kds_[i] = 0.0;
    hw_actuator_hard_limit_active_max_[i] = false;
    hw_actuator_hard_limit_active_min_[i] = false;
    hw_actuator_predicting_[i] = false;
  }

  RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Successfully configured!");

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::CommandInterface>
ControlBoardHardwareInterface::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (auto i = 0u; i < info_.joints.size(); i++) {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_command_positions_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_command_velocities_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_command_efforts_[i]));
    command_interfaces.emplace_back(
        hardware_interface::CommandInterface(info_.joints[i].name, "kp", &hw_command_kps_[i]));
    command_interfaces.emplace_back(
        hardware_interface::CommandInterface(info_.joints[i].name, "kd", &hw_command_kds_[i]));
  }

  return command_interfaces;
}

bool contains_nan(const Eigen::Quaternionf &q) {
  return std::isnan(q.x()) || std::isnan(q.y()) || std::isnan(q.z()) || std::isnan(q.w());
}

hardware_interface::CallbackReturn ControlBoardHardwareInterface::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  // TODO: figure out how to disable if IMU is enabled but not reading properly
  //   if (imu_manager_.enabled && !sample_imu_multiple_attempts(*imu_, /*max_samples=*/500,
  //   /*delay_ms=*/10)) {
  //     RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
  //                  "%sFailed to get a good IMU sample within 500 attempts. Deactivating "
  //                  "motors%s",
  //                  RED_ANSI, RESET_ANSI);
  //     deactivate_motors();
  //     return hardware_interface::CallbackReturn::ERROR;
  //   }

  // Enable actuators. Send the command multiple times to ensure it is received.
  for (int i = 0; i < 10; i++) {
    spi_command_->flags[0] = 1;
    spi_command_->flags[1] = 1;
    spi_command_->flags[2] = 1;
    spi_command_->flags[3] = 1;
    spi_driver_run();
  }
  copy_actuator_states();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Homing
  do_homing();

  RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Successfully activated!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

void ControlBoardHardwareInterface::deactivate_motors() {
  // Disable actuators
  spi_command_->flags[0] = 0;
  spi_command_->flags[1] = 0;
  spi_command_->flags[2] = 0;
  spi_command_->flags[3] = 0;
  spi_driver_run();
}

hardware_interface::CallbackReturn ControlBoardHardwareInterface::on_error(
    [[maybe_unused]] const rclcpp_lifecycle::State &previous_state) {
  deactivate_motors();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ControlBoardHardwareInterface::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  deactivate_motors();
  RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Successfully deactivated!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

bool ControlBoardHardwareInterface::hw_states_contains_nan() {
  return contains_nan(hw_state_positions_) || contains_nan(hw_state_velocities_) ||
         contains_nan(hw_state_efforts_) || contains_nan(hw_state_imu_orientation_) ||
         contains_nan(hw_state_imu_angular_velocity_) ||
         contains_nan(hw_state_imu_linear_acceleration_);
}

hardware_interface::return_type ControlBoardHardwareInterface::read(
    [[maybe_unused]] const rclcpp::Time &time, [[maybe_unused]] const rclcpp::Duration &period) {
  // Write to and read from the actuators
  spi_driver_run();
  copy_actuator_states();

  // Print joint position
  // RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Joint 0: %f",
  // spi_data_->q_abad[1]);

  tf2::Quaternion corrected_quat = tf2::Quaternion::getIdentity();
  tf2::Vector3 angular_velocity(0, 0, 0);
  tf2::Vector3 linear_acceleration(0, 0, 0);

  if (imu_manager_.enabled) {
    // TODO: put this functionality into the imu_manager
    BNO055::Output imu_output = imu_manager_.get_imu_data();
    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();

    imu_packet_timestamp_micros_ = std::chrono::duration_cast<std::chrono::microseconds>(
                                       imu_output.packet_timestamp.time_since_epoch())
                                       .count();
    imu_measurement_timestamp_micros_ = std::chrono::duration_cast<std::chrono::microseconds>(
                                            imu_output.measurement_timestamp.time_since_epoch())
                                            .count();
    imu_time_since_measurement_micros_ = std::chrono::duration_cast<std::chrono::microseconds>(
                                             now - imu_output.measurement_timestamp)
                                             .count();
    imu_time_since_measurement_seconds_ = imu_time_since_measurement_micros_ / 1e6;

    // Print imu timestamps and age
    // RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"),
    //             "Packet: %ld us\tmesurement: %ldus\tage as of read(): %ldus = %fs",
    //             imu_packet_timestamp_micros_, imu_measurement_timestamp_micros_,
    //             imu_time_since_measurement_micros_, imu_time_since_measurement_seconds_);

    // Represent IMU orientation as quaternion
    tf2::Quaternion imu_quat(imu_output.quat.x(), imu_output.quat.y(), imu_output.quat.z(),
                             imu_output.quat.w());

    // Applying the offset to the IMU quaternion
    corrected_quat = imu_quat * imu_manager_.offset_quaternion.inverse();
    corrected_quat.normalize();

    // Rotating the angular velocity
    tf2::Vector3 imu_ang_vel(imu_output.gyro.x(), imu_output.gyro.y(), imu_output.gyro.z());
    angular_velocity = imu_manager_.offset_rotation_matrix * imu_ang_vel;

    // Rotating the linear acceleration
    tf2::Vector3 imu_acc(imu_output.acc.x(), imu_output.acc.y(), imu_output.acc.z());
    linear_acceleration = imu_manager_.offset_rotation_matrix * imu_acc;
  }

  // Updating the state interfaces with corrected values
  hw_state_imu_orientation_[0] = corrected_quat.x();
  hw_state_imu_orientation_[1] = corrected_quat.y();
  hw_state_imu_orientation_[2] = corrected_quat.z();
  hw_state_imu_orientation_[3] = corrected_quat.w();

  hw_state_imu_angular_velocity_[0] = angular_velocity.x();
  hw_state_imu_angular_velocity_[1] = angular_velocity.y();
  hw_state_imu_angular_velocity_[2] = angular_velocity.z();

  hw_state_imu_linear_acceleration_[0] = linear_acceleration.x();
  hw_state_imu_linear_acceleration_[1] = linear_acceleration.y();
  hw_state_imu_linear_acceleration_[2] = linear_acceleration.z();

  // Check if any NaNs in hardware state arrays. Catches IMU issues.
  if (hw_states_contains_nan()) {
    RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
                 "HW state array contained NaN. Deactivating motors");
    std::stringstream ss;
    ss << hw_state_imu_orientation_ << '\n'
       << hw_state_imu_angular_velocity_ << '\n'
       << hw_state_imu_linear_acceleration_ << '\n'
       << hw_state_positions_ << '\n'
       << hw_state_velocities_ << '\n'
       << hw_state_efforts_ << '\n';
    RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"), "%s", ss.str().c_str());
    deactivate_motors();
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type ControlBoardHardwareInterface::write(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
  copy_actuator_commands(true);
  return hardware_interface::return_type::OK;
}

void ControlBoardHardwareInterface::do_homing() {
  RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Homing actuators...");

  int dt_ms = 10;

  std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));
  copy_actuator_commands();
  spi_driver_run();
  copy_actuator_states();
  std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));

  // Set the initial commands
  for (auto i = 0u; i < hw_state_positions_.size(); i++) {
    hw_command_positions_[i] = hw_state_positions_[i];
    hw_command_velocities_[i] = 0.0;
    hw_command_kps_[i] = 0.0;
    hw_command_kds_[i] = 0.0;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));
  copy_actuator_commands();
  spi_driver_run();
  copy_actuator_states();
  std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));

  // Keep track of filtered torques
  std::vector<double> filtered_torques;
  filtered_torques.resize(hw_state_positions_.size(), 0.0);
  double alpha = 0.5;

  // Loop until all actuators are homed
  bool all_homed = false;
  bool all_returned = false;
  while (!all_homed) {
    // Set the homing stage to that of the lowest-stage unhomed actuator
    int homing_stage = 999;
    for (auto i = 0u; i < hw_actuator_homing_stages_.size(); i++) {
      if (!hw_actuator_is_homed_[i] && (hw_actuator_homing_stages_[i] < homing_stage)) {
        homing_stage = hw_actuator_homing_stages_[i];
      }
    }
    // Check if all actuators are homed
    all_homed = true;
    for (auto i = 0u; i < hw_state_positions_.size(); i++) {
      if (homing_stage < hw_actuator_homing_stages_[i]) {
        hw_command_positions_[i] = hw_state_positions_[i];
        hw_command_velocities_[i] = 0.0;
        hw_command_kps_[i] = 0.0;
        hw_command_kds_[i] = 0.0;
        all_homed = false;
        continue;
      } else {
        hw_command_kps_[i] = hw_actuator_homing_kps_[i];
        hw_command_kds_[i] = hw_actuator_homing_kds_[i];
      }
      // std::cout << "Commanded torque: " << filtered_torques[i] << std::endl;
      // std::cout << "Current position: " << hw_state_positions_[i] << std::endl;
      // std::cout << "Commanded position: " << hw_command_positions_[i] << std::endl;
      // WARNING if you ever re-enable active (nonzero-threshold) torque-seeking homing on
      // a joint that also has the copy_actuator_commands() hard end-stop active: this
      // torque estimate is computed from the RAW hw_command_positions_/velocities_
      // members, not from what copy_actuator_commands() actually clamps them to before
      // sending. If the end-stop is holding the joint short of where this loop thinks
      // it commanded, filtered_torques will read a torque that was never actually
      // applied, and "homed" will be declared at the wrong position -- silently shifting
      // this joint's entire safety band by an arbitrary amount for the rest of the
      // session. Currently inert for every joint using this pattern (their
      // homing_torque_threshold is 0.0, so this branch fires on iteration 1 before any
      // meaningful command divergence can accumulate) -- but do not add real
      // torque-threshold homing back to a joint with a tight position_min/max without
      // first fixing this to use the same clamped values copy_actuator_commands() sends.
      if (!hw_actuator_is_homed_[i]) {
        filtered_torques[i] =
            (1.0 - alpha) * filtered_torques[i] +
            alpha * ((hw_command_positions_[i] - hw_state_positions_[i]) * hw_command_kps_[i] +
                     (hw_command_velocities_[i] - hw_state_velocities_[i]) * hw_command_kds_[i]);
        if (std::abs(filtered_torques[i]) >= hw_actuator_homing_torque_thresholds_[i]) {
          // Verify the raw pre-homing reading against a known-good reference (see
          // homing_reference_raw in on_init()) BEFORE it gets snapped to homed_position
          // below -- confirmed empirically (4 boots, legs left untouched) that this
          // sensor reading is absolute/repeatable when the joint is genuinely resting in
          // the gravity-drop pose, and that a disturbed/mis-caught leg reads differently
          // (one boot differed from the other three by up to 33 deg on this same
          // hardware). This is the only check that isn't circular -- see the comment
          // further down for why comparing post-snap values can't catch this.
          if (std::isfinite(hw_actuator_homing_reference_raw_[i])) {
            constexpr double kHomingRawMismatchThreshold = 0.05;  // ~2.9 deg
            const double raw_diff =
                std::abs(hw_state_positions_[i] - hw_actuator_homing_reference_raw_[i]);
            if (raw_diff > kHomingRawMismatchThreshold) {
              RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
                           "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
              RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
                           "!!! HOMING MISMATCH on %s: raw=%.4f, expected=%.4f (diff=%.1f "
                           "deg) -- this joint was likely NOT resting in the correct "
                           "gravity-drop pose at boot. DO NOT TRUST THIS HOMING. !!!",
                           info_.joints[i].name.c_str(), hw_state_positions_[i],
                           hw_actuator_homing_reference_raw_[i],
                           raw_diff * 180.0 / M_PI);
              RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
                           "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            } else {
              RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"),
                          "Homing reference check OK for %s: raw=%.4f (expected %.4f, "
                          "diff=%.1f deg)",
                          info_.joints[i].name.c_str(), hw_state_positions_[i],
                          hw_actuator_homing_reference_raw_[i], raw_diff * 180.0 / M_PI);
            }
          }
          hw_actuator_zero_positions_[i] = hw_state_positions_[i] - hw_actuator_homed_positions_[i];
          // Put hw_state_positions_ in the new (joint) frame in this SAME cycle, not the
          // next copy_actuator_states() call. Review found that without this, the
          // is_homed_ gate on the hard end-stop in copy_actuator_commands() opens one
          // cycle before hw_state_positions_ reflects the offset it now depends on --
          // copy_actuator_commands() runs later this same cycle and would otherwise
          // compare a still-raw encoder value against a joint-frame limit. By
          // definition state - zero == homed_position, so this is exact, not approximate.
          hw_state_positions_[i] = hw_actuator_homed_positions_[i];
          hw_command_positions_[i] = hw_actuator_homed_positions_[i];
          hw_command_velocities_[i] = 0.0;
          hw_actuator_is_homed_[i] = true;
          RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Homed actuator %d", i);
        } else {
          hw_command_positions_[i] += hw_actuator_homing_velocities_[i] * dt_ms * 0.001;
          hw_command_velocities_[i] = hw_actuator_homing_velocities_[i];
          all_homed = false;
        }
      }
    }
    copy_actuator_commands();
    spi_driver_run();
    copy_actuator_states();

    // Sleep for dt
    std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));
  }
  while (!all_returned) {
    all_returned = true;
    for (auto i = 0u; i < hw_state_positions_.size(); i++) {
      if (hw_command_positions_[i] < hw_actuator_post_homing_positions_[i] - 0.01) {
        hw_command_positions_[i] += std::abs(hw_actuator_homing_velocities_[i]) * dt_ms * 0.001;
        hw_command_velocities_[i] = std::abs(hw_actuator_homing_velocities_[i]);
        all_returned = false;
      } else if (hw_command_positions_[i] > hw_actuator_post_homing_positions_[i] + 0.01) {
        hw_command_positions_[i] -= std::abs(hw_actuator_homing_velocities_[i]) * dt_ms * 0.001;
        hw_command_velocities_[i] = -std::abs(hw_actuator_homing_velocities_[i]);
        all_returned = false;
      } else {
        hw_command_positions_[i] = hw_actuator_post_homing_positions_[i];
        hw_command_velocities_[i] = 0.0;
      }
    }
    copy_actuator_commands();
    spi_driver_run();
    copy_actuator_states();

    // Sleep for dt
    std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));
  }

  // Sanity check for the four disarmed-homing knee/foot joints (see components.xacro):
  // their homed reading is NOT an independent measurement of anything -- by construction,
  // zero_position is computed so the reading immediately equals the configured
  // homed_position, so comparing it against that same configured value would always
  // trivially "pass" no matter what pose the leg was actually in. What CAN be checked
  // without that circularity: front_r_3 and back_r_3 (and front_l_3/back_l_3) are the
  // same mechanical design and measured within ~0.1-0.2 deg of each other when
  // hardware-calibrated. A same-side front/back mismatch after homing means one of that
  // pair is NOT resting in the expected gravity-droop pose right now (caught on
  // something, held, resting on a surface, etc.) -- exactly the precondition violation
  // the disarmed-homing design depends on, made visible instead of silent.
  {
    constexpr double kSideMismatchThreshold = 0.0873;  // ~5 deg
    const std::pair<const char *, const char *> side_pairs[] = {
        {"leg_front_r_3", "leg_back_r_3"},
        {"leg_front_l_3", "leg_back_l_3"},
    };
    for (const auto &pair : side_pairs) {
      int idx_a = -1, idx_b = -1;
      for (auto i = 0u; i < info_.joints.size(); i++) {
        if (info_.joints[i].name == pair.first) idx_a = static_cast<int>(i);
        if (info_.joints[i].name == pair.second) idx_b = static_cast<int>(i);
      }
      if (idx_a < 0 || idx_b < 0) continue;  // joints not present on this description; skip
      const double diff = std::abs(hw_state_positions_[idx_a] - hw_state_positions_[idx_b]);
      if (diff > kSideMismatchThreshold) {
        RCLCPP_ERROR(rclcpp::get_logger("ControlBoardHardwareInterface"),
                     "Homing sanity check FAILED: %s (%.4f) and %s (%.4f) differ by %.4f rad "
                     "(%.1f deg), expected ~0. This means one of these legs was likely NOT "
                     "hanging freely at boot -- do not trust the hard position limit on either "
                     "joint in this pair until this is investigated.",
                     pair.first, hw_state_positions_[idx_a], pair.second,
                     hw_state_positions_[idx_b], diff, diff * 180.0 / M_PI);
      } else {
        RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"),
                    "Homing sanity check OK: %s (%.4f) and %s (%.4f) agree within %.1f deg",
                    pair.first, hw_state_positions_[idx_a], pair.second,
                    hw_state_positions_[idx_b], diff * 180.0 / M_PI);
      }
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("ControlBoardHardwareInterface"), "Finished homing!");
}

void ControlBoardHardwareInterface::copy_actuator_commands(bool use_position_limits) {
  // Iterate through the joints
  for (auto i = 0u; i < hw_state_positions_.size(); i++) {
    double cmd_pos = hw_command_positions_[i];
    double cmd_vel = std::clamp(hw_command_velocities_[i], -hw_actuator_velocity_maxs_[i],
                                hw_actuator_velocity_maxs_[i]);
    double cmd_eff = std::clamp(hw_command_efforts_[i], -hw_actuator_effort_maxs_[i],
                                hw_actuator_effort_maxs_[i]);
    double cmd_kp = std::clamp(hw_command_kps_[i], 0.0, hw_actuator_kp_maxs_[i]);
    double cmd_kd = std::clamp(hw_command_kds_[i], 0.0, hw_actuator_kd_maxs_[i]);

    if (use_position_limits && cmd_kp > 0.0) {
      if (cmd_pos < hw_actuator_position_mins_[i]) {
        cmd_pos = hw_actuator_position_mins_[i];
        cmd_vel = std::clamp(cmd_vel, 0.0, hw_actuator_velocity_maxs_[i]);
        cmd_eff = std::clamp(cmd_eff, 0.0, hw_actuator_effort_maxs_[i]);
      } else if (cmd_pos > hw_actuator_position_maxs_[i]) {
        cmd_pos = hw_actuator_position_maxs_[i];
        cmd_vel = std::clamp(cmd_vel, -hw_actuator_velocity_maxs_[i], 0.0);
        cmd_eff = std::clamp(cmd_eff, -hw_actuator_effort_maxs_[i], 0.0);
      }
      cmd_pos = std::clamp(cmd_pos, hw_actuator_position_mins_[i], hw_actuator_position_maxs_[i]);
    }

    // Hard end-stop, unconditional: unlike the block above, this does NOT depend on
    // use_position_limits or cmd_kp > 0.0, and does NOT use position_min/max -- it uses
    // the separate hw_actuator_hard_limit_mins_/maxs_ (see the header comment for why:
    // position_min/max is a soft command clip sized with headroom inside each policy's
    // own commanded envelope, not a real mechanical limit, and is +-infinity here for
    // every joint except the ones that opt in with a finite hard_limit_min/max in the
    // xacro). It exists because the block above only restrains what gets COMMANDED
    // under position control -- it does nothing in pure velocity mode (cmd_kp == 0,
    // e.g. a wheel-style controller), since torque = kp*(cmd_pos-state_pos) +
    // kd*(cmd_vel-state_vel) and kp=0 makes the position term vanish regardless of what
    // cmd_pos is clamped to. A joint whose real hardware self-destructs if driven far
    // enough (no mechanical hard stop, e.g. this board's leg knee joints) needs a limit
    // that holds no matter what commanded the motion -- a future controller in velocity
    // mode, a homing bug, anything.
    //
    // Two independent adversarial reviews (before this ever ran on hardware) each found
    // real problems and are both addressed below:
    //
    // Round 1 found: zeroing cmd_vel alone under kp=0 is a damper, not a stop (cannot
    // hold against so much as gravity, so the joint creeps through indefinitely); and
    // checking only the instantaneous position leaves no braking distance against
    // pipeline latency. Fixed by seizing full position-control authority at the
    // boundary (forcing kp/kd to their maxima) plus a short lookahead so the response
    // starts before the joint physically reaches the line.
    //
    // Round 2 found the round-1 fix introduced three new problems:
    // 1. Reusing position_min/max meant this fired during ordinary in-envelope
    //    operation on 8 of 12 joints (those values sit only ~0.1 rad inside what
    //    deployed policies actually command) -- gain-seizing mid-stride on a
    //    load-bearing leg. Fixed by using the separate, opt-in hard_limit_min/max
    //    instead, which is finite ONLY on the four knee/foot joints that actually need
    //    it.
    // 2. `cmd_pos = boundary` was an unconditional ASSIGNMENT, not a clamp. While still
    //    inside the band but predicted to exceed it, boundary > state_pos, so
    //    kp_max*(boundary-state_pos) is a POSITIVE torque pulling the joint TOWARD the
    //    limit -- an attractor, not a repeller, that only happened to net negative
    //    because of an unrelated kd/lookahead ratio. Fixed: cmd_pos is now clamped
    //    (std::min/std::max against the boundary) instead of assigned, so it can only
    //    ever pull the *commanded* position back toward safety, never push it further.
    // 3. Before a joint is homed, hw_actuator_zero_positions_[i] is still 0, so
    //    hw_state_positions_[i] is the RAW encoder value being compared against a
    //    joint-frame limit -- a frame mismatch that could seize full kp_max/kd_max
    //    authority toward an arbitrary raw-frame target while a leg is supposed to be
    //    hanging free for calibration, with motors already enabled. Fixed by gating
    //    this whole block on hw_actuator_is_homed_[i].
    //
    // Round 2 also found the NaN path didn't actually fail closed (cmd_kp passed
    // through untouched, so a position-mode controller's original unclamped target
    // still got driven at full authority) -- fixed by also zeroing cmd_kp. And flagged
    // that a bare threshold crossing would chatter kp/kd on and off every cycle right
    // at the boundary -- fixed with a release-margin hysteresis latch.
    constexpr double kLookaheadSeconds = 0.02;  // several x the >=2-cycle pipeline latency at 520 Hz
    // The predictive (lookahead) trigger is only meaningful, and only safe, when
    // gated to genuinely fast motion. At this margin (hard_limit sits only a few
    // degrees beyond what the deployed policy's own gait actually commands), a 20ms
    // lookahead at even a normal walking angular velocity covers enough distance to
    // spuriously predict a crossing that will never happen -- fighting a policy that
    // was never actually going to violate the limit. The velocity gate itself is
    // hysteretic (arms at the high threshold, only disarms below the low one) so
    // velocity dithering right at a single threshold can't flicker the prediction
    // on and off every cycle -- round 3 review found exactly that failure mode with
    // a single un-hysteresced threshold.
    // NOTE: both thresholds are chosen conservatively above a rough estimate of
    // normal walking angular velocity for this joint, not measured from a real
    // logged gait -- revisit against actual /joint_states velocity data from a real
    // walking test before trusting them blindly.
    constexpr double kPredictiveVelocityArmThreshold = 8.0;    // rad/s
    constexpr double kPredictiveVelocityDisarmThreshold = 6.0;  // rad/s
    constexpr double kHardLimitReleaseMargin = 0.0349;  // ~2 deg; re-arm only once clearly inside
    const bool state_finite =
        std::isfinite(hw_state_positions_[i]) && std::isfinite(hw_state_velocities_[i]);
    if (!state_finite) {
      // Can't tell which direction is safe without a valid reading. Fail closed: refuse
      // any new motion (including position-mode authority) rather than fail open, which
      // is what would happen if this just fell through to the >=/<= checks below, since
      // NaN compares false to everything.
      cmd_vel = 0.0;
      cmd_eff = 0.0;
      cmd_kp = 0.0;
    } else if (hw_actuator_is_homed_[i]) {
      const double abs_vel = std::abs(hw_state_velocities_[i]);
      const bool predicting = hw_actuator_predicting_[i]
                                   ? (abs_vel >= kPredictiveVelocityDisarmThreshold)
                                   : (abs_vel >= kPredictiveVelocityArmThreshold);
      hw_actuator_predicting_[i] = predicting;
      const double predicted_pos =
          predicting ? hw_state_positions_[i] + hw_state_velocities_[i] * kLookaheadSeconds
                     : hw_state_positions_[i];
      const double hard_max = hw_actuator_hard_limit_maxs_[i];
      const double hard_min = hw_actuator_hard_limit_mins_[i];

      // Instantaneous triggers -- the joint is ACTUALLY at or past the boundary right
      // now. No false-positive risk regardless of velocity; this is the only thing
      // allowed to seize cmd_kp (see round-3 finding below).
      const bool instant_max = hw_state_positions_[i] >= hard_max;
      const bool instant_min = hw_state_positions_[i] <= hard_min;

      // Each direction's latch is updated independently, from the trigger that actually
      // fired for THAT direction -- not derived from current position relative to some
      // midpoint, which could disagree with a predicted-position trigger on the other
      // side of it. (Plain bool, not bool&: std::vector<bool> is bit-packed and its
      // operator[] returns a proxy, not a real reference, so read-compute-write instead
      // of binding a reference to it.) The release check requires BOTH the actual and
      // predicted position to be back inside the release margin, not predicted_pos
      // alone -- round 4 review found that releasing on predicted_pos alone let the
      // latch clear while the joint's ACTUAL position was still past the boundary
      // during a fast retreat (predicted_pos can swing back inside well before actual
      // position does), toggling full protection on and off every other cycle. Requiring
      // both is monotone-safe on approach and on retreat.
      const bool active_max =
          hw_actuator_hard_limit_active_max_[i]
              ? (hw_state_positions_[i] > hard_max - kHardLimitReleaseMargin ||
                 predicted_pos > hard_max - kHardLimitReleaseMargin)
              : (instant_max || predicted_pos >= hard_max);
      hw_actuator_hard_limit_active_max_[i] = active_max;

      const bool active_min =
          hw_actuator_hard_limit_active_min_[i]
              ? (hw_state_positions_[i] < hard_min + kHardLimitReleaseMargin ||
                 predicted_pos < hard_min + kHardLimitReleaseMargin)
              : (instant_min || predicted_pos <= hard_min);
      hw_actuator_hard_limit_active_min_[i] = active_min;

      // Clamp, never assign: this can only pull an out-of-range commanded position back
      // toward the boundary, never push an in-range one further (see round-2 finding #2
      // above). Both branches can't legitimately be active at once (hard_min < hard_max
      // with a real gap between them), but apply independently regardless so a
      // misconfigured band fails toward "commands nothing" rather than an inconsistent
      // combination.
      //
      // cmd_kp is seized ONLY on the instantaneous trigger, not a purely predictive one.
      // Round 3 review found that seizing kp on prediction alone creates a residual
      // attractor: while still inside the band, torque = kp_max*(cmd_pos-state_pos) is
      // POSITIVE whenever the controller's target leads the joint -- true under normal
      // tracking -- and pipeline latency can let that outlast the braking kd term that
      // was supposed to offset it. A purely predictive trigger instead only clamps
      // velocity/effort and raises kd (extra damping, no invented positional pull);
      // full position-hold authority only engages once the joint has genuinely reached
      // the line.
      if (active_max) {
        cmd_pos = std::min(cmd_pos, hard_max);
        cmd_vel = std::min(cmd_vel, 0.0);
        cmd_eff = std::min(cmd_eff, 0.0);
        cmd_kd = hw_actuator_kd_maxs_[i];
        if (instant_max) {
          cmd_kp = hw_actuator_kp_maxs_[i];
        }
      }
      if (active_min) {
        cmd_pos = std::max(cmd_pos, hard_min);
        cmd_vel = std::max(cmd_vel, 0.0);
        cmd_eff = std::max(cmd_eff, 0.0);
        cmd_kd = hw_actuator_kd_maxs_[i];
        if (instant_min) {
          cmd_kp = hw_actuator_kp_maxs_[i];
        }
      }
    }

    cmd_pos += hw_actuator_zero_positions_[i];

    uint can_channel = hw_actuator_can_channels_[i] - 1;
    // ID 1: abad, ID 2: hip, ID 3: knee (not corresponding to the actual joint names, just used
    // to make the Cheetah code send to the CAN IDs we want)
    switch (hw_actuator_can_ids_[i]) {
      case 1:
        spi_command_->q_des_abad[can_channel] = cmd_pos;
        spi_command_->qd_des_abad[can_channel] = cmd_vel;
        spi_command_->kp_abad[can_channel] = cmd_kp;
        spi_command_->kd_abad[can_channel] = cmd_kd;
        spi_command_->tau_abad_ff[can_channel] = cmd_eff;
        break;
      case 2:
        spi_command_->q_des_hip[can_channel] = cmd_pos;
        spi_command_->qd_des_hip[can_channel] = cmd_vel;
        spi_command_->kp_hip[can_channel] = cmd_kp;
        spi_command_->kd_hip[can_channel] = cmd_kd;
        spi_command_->tau_hip_ff[can_channel] = cmd_eff;
        break;
      case 3:
        spi_command_->q_des_knee[can_channel] = cmd_pos;
        spi_command_->qd_des_knee[can_channel] = cmd_vel;
        spi_command_->kp_knee[can_channel] = cmd_kp;
        spi_command_->kd_knee[can_channel] = cmd_kd;
        spi_command_->tau_knee_ff[can_channel] = cmd_eff;
        break;
    }
  }
}

void ControlBoardHardwareInterface::copy_actuator_states() {
  // Iterate through the joints
  for (auto i = 0u; i < hw_state_positions_.size(); i++) {
    float state_pos = hw_state_positions_[i];
    float state_vel = hw_state_velocities_[i];

    uint can_channel = hw_actuator_can_channels_[i] - 1;
    // ID 1: abad, ID 2: hip, ID 3: knee (not corresponding to the actual joint names, just used
    // to make the Cheetah code send to the CAN IDs we want)
    switch (hw_actuator_can_ids_[i]) {
      case 1:
        state_pos = spi_data_->q_abad[can_channel];
        state_vel = spi_data_->qd_abad[can_channel];
        break;
      case 2:
        state_pos = spi_data_->q_hip[can_channel];
        state_vel = spi_data_->qd_hip[can_channel];
        break;
      case 3:
        state_pos = spi_data_->q_knee[can_channel];
        state_vel = spi_data_->qd_knee[can_channel];
        break;
    }
    hw_state_positions_[i] = state_pos - hw_actuator_zero_positions_[i];
    hw_state_velocities_[i] = state_vel;

    // Estimate actuator efforts based on motor driver PD control
    hw_state_efforts_[i] =
        (hw_command_positions_[i] - hw_state_positions_[i]) * hw_command_kps_[i] +
        (hw_command_velocities_[i] - hw_state_velocities_[i]) * hw_command_kds_[i];
  }
}

}  // namespace control_board_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(control_board_hardware_interface::ControlBoardHardwareInterface,
                       hardware_interface::SystemInterface)
