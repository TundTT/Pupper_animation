#include "neural_controller/keyframe_controller.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace neural_controller {
controller_interface::CallbackReturn KeyframeController::on_init() {
  try {
    param_listener_=std::make_shared<ParamListener>(get_node());params_=param_listener_->get_params();
    if(!check_param_vector_size())throw std::runtime_error("Invalid joint configuration sizes");
    if(get_update_rate()!=520||params_.repeat_action!=1||!params_.use_imu||params_.gain_multiplier!=1||params_.estop_kd!=1||params_.max_body_angle!=.6)
      throw std::runtime_error("Keyframes require 520 Hz, IMU, gain multiplier 1 and stop damping 1");
    const std::array<std::string,4> names{"front_r","front_l","back_r","back_l"};
    for(int i=0;i<12;++i){
      const bool wheel=i%3==2;
      if(params_.joint_names[i]!="leg_"+names[i/3]+"_"+std::to_string(i%3+1)||
         params_.action_types[i]!=(wheel ? "velocity":"position")||
         !std::isfinite(params_.kps[i])||!std::isfinite(params_.kds[i])||
         std::abs(params_.kps[i]-(wheel ? 0.:5.))>1e-6||
         std::abs(params_.kds[i]-(wheel ? .35:.25))>1e-6)
        throw std::runtime_error("Keyframe joint order, modes or gains differ from audited model");
      if(!wheel){const int a=2*(i/3)+i%3;
        if(params_.joint_lower_limits[i]!=keyframe_align::Geometry::low[a]||params_.joint_upper_limits[i]!=keyframe_align::Geometry::high[a])
          throw std::runtime_error("Proximal limits differ from keyframe contract");
      }else if(params_.init_kps[i]!=0||params_.joint_lower_limits[i]>-100||params_.joint_upper_limits[i]<100)
        throw std::runtime_error("Continuous wheel profile required");
    }
    nlohmann::json j;std::ifstream input(params_.model_path);input>>j;
    const std::array<std::string,15> fields{"entry_seconds","shift_seconds","lift_seconds","land_seconds","recenter_seconds",
      "attempt_timeout_seconds","wheel_kp","wheel_kd","wheel_ki","wheel_integral_limit","wheel_speed_limit",
      "wheel_acceleration_limit","abduction_speed_limit","hip_speed_limit","joint_acceleration_limit"};
    for(int i=0;i<15;++i)keyframes_.config.values[i]=j.at(fields[i]).get<double>();
    keyframes_.config.poses=j.at("poses").get<std::array<keyframe_align::V8,4>>();keyframes_.config.validate();
    behavior_="keyframe_align";single_observation_size_=6;params_.observation_history=1;
    command_states_={"stand","front_l","front_r","back_r","back_l"};num_commands_=5;
    return controller_interface::CallbackReturn::SUCCESS;
  }catch(const std::exception& e){RCLCPP_ERROR(get_node()->get_logger(),"Keyframe configuration rejected: %s",e.what());return controller_interface::CallbackReturn::ERROR;}
}

void KeyframeController::brake(){
  for(auto& interface:command_interfaces_)interface.set_value(0.);
  for(const auto& name:params_.joint_names)command_interfaces_map_.at(name).at("kd").get().set_value(params_.estop_kd);
}

bool KeyframeController::read_sensors(keyframe_align::V12& q,keyframe_align::V12& qd,keyframe_align::V3& w,keyframe_align::V3& g){
  for(int i=0;i<12;++i){
    q[i]=state_interfaces_map_.at(params_.joint_names[i]).at("position").get().get_value();
    qd[i]=state_interfaces_map_.at(params_.joint_names[i]).at("velocity").get().get_value();
    if(!std::isfinite(q[i])||!std::isfinite(qd[i]))return false;
  }
  auto& imu=state_interfaces_map_.at("imu_sensor");
  const std::array<std::string,3> axes{"x","y","z"};
  for(int i=0;i<3;++i){w[i]=imu.at("angular_velocity."+axes[i]).get().get_value();if(!std::isfinite(w[i]))return false;}
  tf2::Quaternion quat(imu.at("orientation.x").get().get_value(),imu.at("orientation.y").get().get_value(),
    imu.at("orientation.z").get().get_value(),imu.at("orientation.w").get().get_value());
  if(!std::isfinite(quat.length2())||std::abs(quat.length2()-1)>.02)return false;
  quat.normalize();auto gravity=tf2::Matrix3x3(quat).transpose()*tf2::Vector3(0,0,-1);
  g={gravity.x(),gravity.y(),gravity.z()};
  auto age=imu.find("time_since_measurement_seconds");
  if(age==imu.end())return !params_.calibration_required; // Fake simulation interfaces only.
  double seconds=age->second.get().get_value();
  return std::isfinite(seconds)&&seconds>=0&&seconds<=.1;
}

controller_interface::CallbackReturn KeyframeController::on_activate(const rclcpp_lifecycle::State& state){
  try {
    auto result=NeuralController::on_activate(state);if(result!=controller_interface::CallbackReturn::SUCCESS)return result;
    // Resolve all required command/state interfaces before the realtime loop.
    for(const auto& name:params_.joint_names)for(const auto& field:{"position","velocity","effort","kp","kd"})
      command_interfaces_map_.at(name).at(field);
    keyframe_align::V12 q{},qd{};keyframe_align::V3 w{},g{};
    if(!read_sensors(q,qd,w,g)||-g[2]<std::cos(.6))throw std::runtime_error("Invalid/stale activation sensors");
    keyframes_.reset(q,startup_calibration_.wheel_home);last_update_seconds_=-1;status_count_=0;
    alignment_status_publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/alignment_status",1);
    rt_alignment_status_publisher_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(alignment_status_publisher_);
    rt_alignment_status_publisher_->msg_.data.resize(25);
    brake();return controller_interface::CallbackReturn::SUCCESS;
  }catch(const std::exception& e){
    RCLCPP_ERROR(get_node()->get_logger(),"Keyframe activation rejected: %s",e.what());
    estop_active_=true;
    // Failure may precede complete interface acquisition; zero any claimed outputs.
    for(auto& interface:command_interfaces_)interface.set_value(interface.get_interface_name()=="kd" ? 1.:0.);
    return controller_interface::CallbackReturn::ERROR;
  }
}

controller_interface::return_type KeyframeController::update(const rclcpp::Time& time,const rclcpp::Duration& period){
  keyframe_align::V12 q{},qd{};keyframe_align::V3 w{},g{};
  const double now=time.seconds();
  if((last_update_seconds_>=0&&(now<=last_update_seconds_||now-last_update_seconds_>.04))||!read_sensors(q,qd,w,g))estop_active_=true;
  last_update_seconds_=now;
  if(estop_active_){brake();publish_status(keyframes_.step(period.seconds(),0,q,qd,w,g,true));return controller_interface::return_type::OK;}
  auto command=rt_leg_lift_command_ptr_.readFromRT();if(command&&command->get())command_index_=command->get()->data;
  const auto o=keyframes_.step(period.seconds(),command_index_,q,qd,w,g);
  if(!o.authority){estop_active_=true;brake();publish_status(o);return controller_interface::return_type::OK;}
  for(int i=0;i<12;++i){auto& joint=command_interfaces_map_.at(params_.joint_names[i]);const bool wheel=i%3==2;
    joint.at("position").get().set_value(wheel ? 0.:o.position[2*(i/3)+i%3]);
    joint.at("velocity").get().set_value(wheel ? o.wheel[i/3]:0.);
    joint.at("effort").get().set_value(0.);joint.at("kp").get().set_value(params_.kps[i]);joint.at("kd").get().set_value(params_.kds[i]);
  }
  publish_status(o);
  return controller_interface::return_type::OK;
}

void KeyframeController::publish_status(const keyframe_align::Output& o){
  if(++status_count_%26==0&&rt_alignment_status_publisher_->trylock()){
    auto& d=rt_alignment_status_publisher_->msg_.data;
    std::copy(o.position.begin(),o.position.end(),d.begin());std::copy(o.wheel.begin(),o.wheel.end(),d.begin()+8);
    d[12]=o.phase;d[13]=o.active;d[14]=o.completed;d[15]=o.blocked;d[16]=o.timeout;
    std::copy(o.margins.begin(),o.margins.end(),d.begin()+17);d[20]=o.error;d[21]=o.up_seconds;d[22]=o.integral;d[23]=o.authority;d[24]=o.failed;
    rt_alignment_status_publisher_->unlockAndPublish();
  }
}
}
PLUGINLIB_EXPORT_CLASS(neural_controller::KeyframeController,controller_interface::ControllerInterface)
