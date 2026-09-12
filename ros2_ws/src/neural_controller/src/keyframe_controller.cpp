#include "neural_controller/keyframe_controller.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace neural_controller {
controller_interface::CallbackReturn KeyframeController::on_init() {
  try {
    param_listener_=std::make_shared<ParamListener>(get_node());params_=param_listener_->get_params();
    if(!check_param_vector_size())throw std::runtime_error("Invalid joint configuration sizes");
    if(get_update_rate()!=520||params_.repeat_action!=1||!params_.use_imu||params_.gain_multiplier!=1||params_.estop_kd!=0||params_.max_body_angle!=.6)
      throw std::runtime_error("Keyframes require 520 Hz, IMU, gain multiplier 1 and zero stop gains");
    const std::array<std::string,4> names{"front_r","front_l","back_r","back_l"};
    for(int i=0;i<12;++i){
      const bool wheel=i%3==2;
      if(params_.joint_names[i]!="leg_"+names[i/3]+"_"+std::to_string(i%3+1)||
         params_.action_types[i]!="position"||
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
    if(j.at("wheel_control_mode")!="position_pd")throw std::runtime_error("Position-PD keyframe config required");
    const std::array<std::string,15> fields{"entry_seconds","shift_seconds","lift_seconds","land_seconds","recenter_seconds",
      "attempt_timeout_seconds","","","","","wheel_speed_limit",
      "wheel_acceleration_limit","abduction_speed_limit","hip_speed_limit","joint_acceleration_limit"};
    for(int i=0;i<15;++i)if(!fields[i].empty())keyframes_.config.values[i]=j.at(fields[i]).get<double>();
    keyframes_.config.rotation_floor_clearance_m=j.at("rotation_floor_clearance_m").get<double>();
    keyframes_.config.wheel_position_kp=j.at("wheel_position_kp").get<double>();
    keyframes_.config.wheel_position_kd=j.at("wheel_position_kd").get<double>();
    keyframes_.config.alignment_angle_tolerance_rad=j.at("alignment_angle_tolerance_rad").get<double>();
    keyframes_.config.landing_angle_tolerance_rad=j.at("landing_angle_tolerance_rad").get<double>();
    keyframes_.config.alignment_speed_tolerance_rad_s=j.at("alignment_speed_tolerance_rad_s").get<double>();
    keyframes_.config.alignment_settle_seconds=j.at("alignment_settle_seconds").get<double>();
    keyframes_.config.hold_error_limit_rad=j.at("hold_error_limit_rad").get<double>();
    keyframes_.config.poses=j.at("poses").get<std::array<keyframe_align::V8,4>>();keyframes_.config.validate();
    behavior_="keyframe_align";single_observation_size_=6;params_.observation_history=1;
    command_states_={"stand","front_l","front_r","back_r","back_l"};num_commands_=5;
    motor_commands_publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/motor_commands",1);
    rt_motor_commands_publisher_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(motor_commands_publisher_);
    rt_motor_commands_publisher_->msg_.data.resize(60);
    // Report from the executor, not the realtime update callback.
    fault_timer_=get_node()->create_wall_timer(std::chrono::milliseconds(100),[this]() {
      const int code=fault_code_.load();
      if(code!=reported_fault_ && code!=NONE){
        const char* reasons[]={"none","clock gap/reversal","invalid update period","invalid encoders","invalid IMU","stale/missing IMU age","excessive tilt","invalid command/core fault","operator stop","activation failure"};
        RCLCPP_ERROR(get_node()->get_logger(),"KEYFRAME FAULT %d: %s; zero torque requested, no holding support. Lifecycle reactivation required.",code,reasons[code]);
      }
      reported_fault_=code;
    });
    return controller_interface::CallbackReturn::SUCCESS;
  }catch(const std::exception& e){RCLCPP_ERROR(get_node()->get_logger(),"Keyframe configuration rejected: %s",e.what());return controller_interface::CallbackReturn::ERROR;}
}

void KeyframeController::release_torque(){
  for(auto& interface:command_interfaces_)interface.set_value(0.);
}
void KeyframeController::latch_fault(int code){int expected=NONE;fault_code_.compare_exchange_strong(expected,code);estop_active_=true;release_torque();}
controller_interface::CallbackReturn KeyframeController::on_error(const rclcpp_lifecycle::State&){latch_fault(ACTIVATION);return controller_interface::CallbackReturn::SUCCESS;}

bool KeyframeController::read_sensors(keyframe_align::V12& q,keyframe_align::V12& qd,keyframe_align::V3& w,keyframe_align::V3& g){
  sensor_fault_=ENCODERS;
  for(int i=0;i<12;++i){
    q[i]=state_interfaces_map_.at(params_.joint_names[i]).at("position").get().get_value();
    qd[i]=state_interfaces_map_.at(params_.joint_names[i]).at("velocity").get().get_value();
    if(!std::isfinite(q[i])||!std::isfinite(qd[i]))return false;
  }
  auto& imu=state_interfaces_map_.at("imu_sensor");
  sensor_fault_=IMU;
  const std::array<std::string,3> axes{"x","y","z"};
  for(int i=0;i<3;++i){w[i]=imu.at("angular_velocity."+axes[i]).get().get_value();if(!std::isfinite(w[i]))return false;}
  tf2::Quaternion quat(imu.at("orientation.x").get().get_value(),imu.at("orientation.y").get().get_value(),
    imu.at("orientation.z").get().get_value(),imu.at("orientation.w").get().get_value());
  if(!std::isfinite(quat.length2())||std::abs(quat.length2()-1)>.02)return false;
  quat.normalize();auto gravity=tf2::Matrix3x3(quat).transpose()*tf2::Vector3(0,0,-1);
  g={gravity.x(),gravity.y(),gravity.z()};
  sensor_fault_=IMU_AGE;last_imu_age_=-1;
  auto age=imu.find("time_since_measurement_seconds");
  if(age==imu.end())return !params_.calibration_required; // Fake simulation interfaces only.
  double seconds=age->second.get().get_value();
  last_imu_age_=seconds;
  return std::isfinite(seconds)&&seconds>=0&&seconds<=.1;
}

controller_interface::CallbackReturn KeyframeController::on_activate(const rclcpp_lifecycle::State& state){
  release_torque();
  try {
    auto result=NeuralController::on_activate(state);if(result!=controller_interface::CallbackReturn::SUCCESS){latch_fault(ACTIVATION);return result;}
    // Resolve all required command/state interfaces before the realtime loop.
    for(const auto& name:params_.joint_names)for(const auto& field:{"position","velocity","effort","kp","kd"})
      command_interfaces_map_.at(name).at(field);
    keyframe_align::V12 q{},qd{};keyframe_align::V3 w{},g{};
    if(!read_sensors(q,qd,w,g)||-g[2]<std::cos(.6))throw std::runtime_error("Invalid/stale activation sensors");
    keyframes_.reset(q,startup_calibration_.wheel_home);last_update_seconds_=-1;status_count_=0;
    fault_code_=NONE;last_period_=0;
    alignment_status_publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/alignment_status",1);
    rt_alignment_status_publisher_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(alignment_status_publisher_);
    rt_alignment_status_publisher_->msg_.data.resize(28);
    release_torque();return controller_interface::CallbackReturn::SUCCESS;
  }catch(const std::exception& e){
    RCLCPP_ERROR(get_node()->get_logger(),"Keyframe activation rejected: %s",e.what());
    latch_fault(ACTIVATION);
    return controller_interface::CallbackReturn::ERROR;
  }
}

controller_interface::return_type KeyframeController::update(const rclcpp::Time& time,const rclcpp::Duration& period){
  keyframe_align::V12 q{},qd{};keyframe_align::V3 w{},g{};
  const double now=time.seconds();
  const bool first=last_update_seconds_<0;
  last_period_=period.seconds();
  if(!std::isfinite(now)||now<0||(!first&&(now<=last_update_seconds_||now-last_update_seconds_>.04)))latch_fault(CLOCK);
  if(!read_sensors(q,qd,w,g))latch_fault(sensor_fault_);
  else if(-g[2]<std::cos(.6))latch_fault(TILT_FAULT);
  if(!std::isfinite(last_period_)||last_period_<0||last_period_>.04||(!first&&last_period_==0))latch_fault(PERIOD);
  last_update_seconds_=now;
  if(estop_active_){latch_fault(OPERATOR_STOP);publish_status(keyframes_.step(period.seconds(),0,q,qd,w,g,true));return controller_interface::return_type::OK;}
  // Jazzy starts with period=0. Establish the clock without advancing the
  // trajectory or applying torque. Only this first zero period is accepted.
  if(first&&last_period_==0){release_torque();auto o=keyframes_.output;o.phase=keyframe_align::ENTRY;publish_status(o);return controller_interface::return_type::OK;}
  auto command=rt_leg_lift_command_ptr_.readFromRT();if(command&&command->get())command_index_=command->get()->data;
  const auto o=keyframes_.step(period.seconds(),command_index_,q,qd,w,g);
  if(!o.authority){latch_fault(COMMAND);publish_status(o);return controller_interface::return_type::OK;}
  for(int i=0;i<12;++i){auto& joint=command_interfaces_map_.at(params_.joint_names[i]);const bool wheel=i%3==2;
    joint.at("position").get().set_value(wheel ? o.wheel_position[i/3]:o.position[2*(i/3)+i%3]);
    joint.at("velocity").get().set_value(0.);
    joint.at("effort").get().set_value(0.);joint.at("kp").get().set_value(wheel ? keyframes_.config.wheel_position_kp:params_.kps[i]);joint.at("kd").get().set_value(wheel ? keyframes_.config.wheel_position_kd:params_.kds[i]);
  }
  publish_status(o);
  return controller_interface::return_type::OK;
}

void KeyframeController::publish_status(const keyframe_align::Output& o){
  if(++status_count_%26!=0)return;
  if(rt_motor_commands_publisher_->trylock()){
    auto& values=rt_motor_commands_publisher_->msg_.data;
    const std::array<const char*,5> fields{"position","velocity","effort","kp","kd"};
    for(int i=0;i<12;++i)for(int j=0;j<5;++j)values[5*i+j]=command_interfaces_map_.at(params_.joint_names[i]).at(fields[j]).get().get_value();
    rt_motor_commands_publisher_->unlockAndPublish();
  }
  if(rt_alignment_status_publisher_->trylock()){
    auto& d=rt_alignment_status_publisher_->msg_.data;
    std::copy(o.position.begin(),o.position.end(),d.begin());std::copy(o.wheel.begin(),o.wheel.end(),d.begin()+8);
    d[12]=o.phase;d[13]=o.active;d[14]=o.completed;d[15]=o.blocked;d[16]=o.timeout;
    std::copy(o.margins.begin(),o.margins.end(),d.begin()+17);d[20]=o.error;d[21]=o.up_seconds;d[22]=o.integral;d[23]=o.authority;d[24]=o.failed;
    d[25]=fault_code_.load();d[26]=last_period_;d[27]=last_imu_age_;
    rt_alignment_status_publisher_->unlockAndPublish();
  }
}
}
PLUGINLIB_EXPORT_CLASS(neural_controller::KeyframeController,controller_interface::ControllerInterface)
