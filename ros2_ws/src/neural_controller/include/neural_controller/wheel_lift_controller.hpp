#pragma once
#include "neural_controller/neural_controller.hpp"
#include "neural_controller/wheel_lift/core.hpp"
namespace neural_controller {
class WheelLiftController:public NeuralController {
 public:
 controller_interface::CallbackReturn on_init()override;
 controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&)override;
 controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&)override;
 controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&)override;
 controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&)override;
 protected:
 wheel_lift::Core lift_;
 std::shared_ptr<std_msgs::msg::Int32> last_request_;
 double last_update_=-1,imu_age_=-1;
 wheel_lift::ActorClock actor_clock_;
 unsigned accepted_=0,rejected_=0;
 std::atomic<int> fault_{0};
 rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr status_pub_,motor_pub_;
 std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> rt_status_,rt_motors_;
 rclcpp::TimerBase::SharedPtr fault_timer_;
 int reported_fault_=0;
 bool sensors(wheel_lift::Sensors&);
 void stop(int code);
 void publish(double dt);
};
}
