#include "neural_controller/leg_to_wheel_controller.hpp"
#include "neural_controller/walking_frame.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace neural_controller {
controller_interface::CallbackReturn LegToWheelController::on_init() {
  try {
    param_listener_=std::make_shared<ParamListener>(get_node());params_=param_listener_->get_params();
    if(!check_param_vector_size() || !params_.calibration_required || get_update_rate()!=520 ||
       params_.observation_history!=4 || !params_.use_imu || params_.imu_sensor_name!="imu_sensor" ||
       params_.gain_multiplier!=1 || params_.estop_kd!=0 || params_.init_duration!=0 ||
       params_.fade_in_duration!=0 || params_.max_body_angle!=.6)
      throw std::runtime_error("Manual leg-to-wheel requires calibrated 520 Hz execution and unchanged gains");
    policy_=std::make_unique<leg_to_wheel::Policy>(params_.model_path);
    nlohmann::json j;std::ifstream input(params_.model_path);input>>j;
    if(j.at("provenance").at("checkpoint_sha256")!="62097f09eaa33d8f969be0daefe111c781caa8257d3e3b8f1ec1935146f54164" ||
       j.at("ctrl_dt")!=.02 || j.at("joint_names")!=params_.joint_names)
      throw std::runtime_error("Wrong selected leg-to-wheel export");
    const std::array<std::string,4> legs{"front_r","front_l","back_r","back_l"};
    for(int i=0;i<12;++i) {
      if(params_.joint_names[i]!="leg_"+legs[i/3]+"_"+std::to_string(i%3+1) ||
         params_.action_types[i]!="position" || j.at("action_types")[i]!="position" ||
         params_.default_joint_pos[i]!=policy_->home[i] || params_.action_scales[i]!=policy_->scale[i] ||
         params_.kps[i]!=policy_->kp[i] || params_.kds[i]!=policy_->kd[i] ||
         params_.init_kps[i]!=policy_->kp[i] || params_.init_kds[i]!=policy_->kd[i] ||
         params_.joint_lower_limits[i]!=policy_->lower[i] || params_.joint_upper_limits[i]!=policy_->upper[i])
        throw std::runtime_error("Joint mapping or actuator contract mismatch");
    }
    behavior_="leg_to_wheel_manual";policy_action_size_=12;single_observation_size_=35;
    return controller_interface::CallbackReturn::SUCCESS;
  } catch(const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),"Leg-to-wheel rejected: %s",e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
}

bool LegToWheelController::sensors() {
  for(int i=0;i<12;++i) {
    auto& s=state_interfaces_map_.at(params_.joint_names[i]);
    q_[i]=s.at("position").get().get_value();qd_[i]=s.at("velocity").get().get_value();
    if(!std::isfinite(q_[i]) || !std::isfinite(qd_[i]))return false;
  }
  auto& imu=state_interfaces_map_.at("imu_sensor");
  const std::array<std::string,3> axes{"x","y","z"};
  for(int i=0;i<3;++i) {
    omega_[i]=imu.at("angular_velocity."+axes[i]).get().get_value();
    if(!std::isfinite(omega_[i]))return false;
  }
  tf2::Quaternion quat(imu.at("orientation.x").get().get_value(),imu.at("orientation.y").get().get_value(),
                       imu.at("orientation.z").get().get_value(),imu.at("orientation.w").get().get_value());
  if(!std::isfinite(quat.length2()) || std::abs(quat.length2()-1)>.02)return false;
  quat.normalize();auto g=tf2::Matrix3x3(quat).transpose()*tf2::Vector3(0,0,-1);gravity_={g.x(),g.y(),g.z()};
  imu_age_=imu.at("time_since_measurement_seconds").get().get_value();
  return std::isfinite(imu_age_) && imu_age_>=0 && imu_age_<=.1;
}

controller_interface::CallbackReturn LegToWheelController::on_activate(const rclcpp_lifecycle::State& state) {
  try {
    auto result=NeuralController::on_activate(state);
    if(result!=controller_interface::CallbackReturn::SUCCESS){stop(6);return result;}
    for(const auto& name:params_.joint_names)
      for(const auto* field:{"position","velocity","effort","kp","kd"})command_interfaces_map_.at(name).at(field);
    if(!sensors() || -gravity_[2]<std::cos(.13962634015954636))
      throw std::runtime_error("Fresh sensors and level entry required");
    for(double speed:qd_)if(std::abs(speed)>.15)throw std::runtime_error("Stationary entry required");
    // Startup marks define tips-down, as for walking. Preserve the nearest full
    // encoder turn for the entire activation; never wrap individual targets.
    leg_to_wheel::Joints reference{};
    for(int k=0;k<4;++k)reference[3*k+2]=startup_calibration_.wheel_home[k]-policy_->home[3*k+2];
    encoder_offset_=walking_offsets(q_,policy_->home,reference);
    target_=q_;policy_->reset();sequence_={};sequence_.request(1);
    last_request_.reset();rt_leg_lift_command_ptr_.writeFromNonRT(nullptr);
    leg_lift_command_subscriber_=get_node()->create_subscription<std_msgs::msg::Int32>(
      "/leg_to_wheel/advance",rclcpp::QoS(1).durability_volatile(),
      [this](std_msgs::msg::Int32::SharedPtr msg){rt_leg_lift_command_ptr_.writeFromNonRT(msg);});
    status_pub_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/manual_status",1);
    rt_status_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(status_pub_);
    rt_status_->msg_.data.resize(10);
    motor_pub_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/motor_commands",1);
    rt_motors_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(motor_pub_);
    rt_motors_->msg_.data.resize(60);
    rt_observation_publisher_->msg_.data.resize(140);rt_policy_output_publisher_->msg_.data.resize(12);
    rt_position_command_publisher_->msg_.data.resize(12);
    last_update_=-1;elapsed_=0;accepted_=1;rejected_=0;fault_=0;estop_active_=false;
    return controller_interface::CallbackReturn::SUCCESS;
  } catch(const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),"Leg-to-wheel activation rejected: %s",e.what());stop(6);
    return controller_interface::CallbackReturn::ERROR;
  }
}

void LegToWheelController::stop(int code) {
  int expected=0;fault_.compare_exchange_strong(expected,code);estop_active_=true;
  for(auto& x:command_interfaces_)x.set_value(0.);
}
controller_interface::CallbackReturn LegToWheelController::on_deactivate(const rclcpp_lifecycle::State& state) {
  last_request_.reset();leg_lift_command_subscriber_.reset();return NeuralController::on_deactivate(state);
}
controller_interface::CallbackReturn LegToWheelController::on_error(const rclcpp_lifecycle::State&) {
  stop(4);return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type LegToWheelController::update(const rclcpp::Time& time,const rclcpp::Duration& period) {
  const double now=time.seconds(),dt=period.seconds();const bool first=last_update_<0;
  try {
    if(estop_active_ || fault_){stop(5);publish(dt);return controller_interface::return_type::OK;}
    if(!std::isfinite(now) || now<0 || !std::isfinite(dt) || dt<0 || dt>.01 ||
       (!first && (dt==0 || now<=last_update_ || now-last_update_>.01))) {
      stop(1);publish(dt);return controller_interface::return_type::OK;
    }
    last_update_=now;
    if(!sensors()){stop(2);publish(dt);return controller_interface::return_type::OK;}
    if(-gravity_[2]<std::cos(.6)){stop(3);publish(dt);return controller_interface::return_type::OK;}
    if(first && dt==0)return controller_interface::return_type::OK;
    elapsed_+=dt;
    const bool inference=first || elapsed_+1e-12>=.02;
    if(inference) {
      elapsed_=first?0:elapsed_-.02;
      // Consume at most one event per actor tick, so every accepted lift/lower
      // transition is observed by the trained history and the lowering filter.
      auto ptr=rt_leg_lift_command_ptr_.readFromRT();
      if(ptr && *ptr && *ptr!=last_request_) {
        last_request_=*ptr;
        if(sequence_.request(last_request_->data))++accepted_;else ++rejected_;
      }
      auto q_model=q_;for(int i=0;i<12;++i)q_model[i]-=encoder_offset_[i];
      const auto out=policy_->step_manual(omega_,gravity_,q_model,sequence_.command());
      for(int i=0;i<12;++i)target_[i]=out.position_target[i]+encoder_offset_[i];
      if(rt_observation_publisher_->trylock()) {
        std::copy(policy_->observation.begin(),policy_->observation.end(),rt_observation_publisher_->msg_.data.begin());
        rt_observation_publisher_->unlockAndPublish();
      }
      if(rt_policy_output_publisher_->trylock()) {
        std::copy(out.raw_action.begin(),out.raw_action.end(),rt_policy_output_publisher_->msg_.data.begin());
        rt_policy_output_publisher_->unlockAndPublish();
      }
    }
    for(int i=0;i<12;++i) {
      if(!std::isfinite(target_[i]) || (i%3==2 && std::abs(target_[i])>=1000))
        throw std::runtime_error("Invalid mapped command");
      auto& j=command_interfaces_map_.at(params_.joint_names[i]);
      j.at("position").get().set_value(target_[i]);j.at("velocity").get().set_value(0.);
      j.at("effort").get().set_value(0.);j.at("kp").get().set_value(policy_->kp[i]);j.at("kd").get().set_value(policy_->kd[i]);
    }
    if(inference)publish(dt);
  } catch(const std::exception&) {stop(4);publish(dt);}
  return controller_interface::return_type::OK;
}

void LegToWheelController::publish(double dt) {
  if(rt_status_ && rt_status_->trylock()) {
    auto& d=rt_status_->msg_.data;
    d[0]=sequence_.step;d[1]=sequence_.command();d[2]=sequence_.leg();d[3]=sequence_.lower_requested();
    d[4]=sequence_.step==8;d[5]=fault_.load();d[6]=accepted_;d[7]=rejected_;d[8]=imu_age_;d[9]=dt;
    rt_status_->unlockAndPublish();
  }
  if(rt_motors_ && rt_motors_->trylock()) {
    const std::array<const char*,5> fields{"position","velocity","effort","kp","kd"};
    for(int i=0;i<12;++i)for(int j=0;j<5;++j)
      rt_motors_->msg_.data[5*i+j]=command_interfaces_map_.at(params_.joint_names[i]).at(fields[j]).get().get_value();
    rt_motors_->unlockAndPublish();
  }
}
}
PLUGINLIB_EXPORT_CLASS(neural_controller::LegToWheelController,controller_interface::ControllerInterface)
