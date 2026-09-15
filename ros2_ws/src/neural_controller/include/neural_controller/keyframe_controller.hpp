#pragma once
#include "neural_controller/neural_controller.hpp"
#include "neural_controller/keyframe_align/controller.hpp"

namespace neural_controller {
// Uses the existing lifecycle calibration, interface mapping and stop plumbing.
// No neural model is loaded or evaluated by this plugin.
class KeyframeController : public NeuralController {
 public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;
  controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&) override;
  controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&) override;
 protected:
  keyframe_align::Controller keyframes_;
  double last_update_seconds_=-1;
  int status_count_=0;
  enum Fault { NONE=0, CLOCK=1, PERIOD=2, ENCODERS=3, IMU=4, IMU_AGE=5, TILT_FAULT=6, COMMAND=7, OPERATOR_STOP=8, ACTIVATION=9 };
  std::atomic<int> fault_code_{NONE};
  int sensor_fault_=NONE;
  int reported_fault_=NONE; // Executor timer only.
  double last_period_=0, last_imu_age_=-1;
  rclcpp::TimerBase::SharedPtr fault_timer_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr motor_commands_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> rt_motor_commands_publisher_;
  bool read_sensors(keyframe_align::V12&,keyframe_align::V12&,keyframe_align::V3&,keyframe_align::V3&);
  void release_torque();
  void latch_fault(int code);
  void publish_status(const keyframe_align::Output&);
};
}
