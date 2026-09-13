#pragma once
#include "neural_controller/keyframe_controller.hpp"
#include "neural_controller/hub_roll.hpp"
#include "sensor_msgs/msg/joy.hpp"

namespace neural_controller {
// Reuse audited encoder/IMU access and the calibration lifecycle, not alignment
// geometry or its wheel PID. Suspended hub-only test holds measured proximal angles.
class HubRollController : public KeyframeController {
 public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override;
  controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&) override;
  controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&) override;
 protected:
  hub_roll::Controller triangle_;
  std::shared_ptr<std_msgs::msg::Int32> consumed_;
  std::string plan_hash_;
  std::atomic<int64_t> joy_receipt_ns_{0};
  std::atomic<bool> joy_stop_held_{false};
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr triangle_joy_;
  bool joy_ready() const;
  void receive_joy(const sensor_msgs::msg::Joy& msg);
  void publish_triangle();
};
}
