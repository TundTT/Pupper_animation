#pragma once
#include "neural_controller/neural_controller.hpp"
#include "neural_controller/leg_to_wheel/policy.hpp"

namespace neural_controller {
class LegToWheelController : public NeuralController {
 public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override;
  controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&) override;
  controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&) override;
 protected:
  std::unique_ptr<leg_to_wheel::Policy> policy_;
  leg_to_wheel::ManualSequencer sequence_;
  leg_to_wheel::Joints target_{},q_{},qd_{};
  leg_to_wheel::Vec3 omega_{},gravity_{};
  std::shared_ptr<std_msgs::msg::Int32> last_request_;
  double last_update_=-1,elapsed_=0,imu_age_=0;
  unsigned accepted_=0,rejected_=0;
  std::atomic<int> fault_{0};
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr status_pub_,motor_pub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> rt_status_,rt_motors_;
  bool sensors();
  void stop(int code);
  void publish(double dt);
};
}
