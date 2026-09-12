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
  controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&) override;
 protected:
  keyframe_align::Controller keyframes_;
  double last_update_seconds_=-1;
  int status_count_=0;
  bool read_sensors(keyframe_align::V12&,keyframe_align::V12&,keyframe_align::V3&,keyframe_align::V3&);
  void brake();
  void publish_status(const keyframe_align::Output&);
};
}
