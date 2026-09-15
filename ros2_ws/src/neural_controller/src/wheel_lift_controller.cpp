#include "neural_controller/wheel_lift_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <fstream>
namespace neural_controller {
controller_interface::CallbackReturn WheelLiftController::on_init(){
 try{
  param_listener_=std::make_shared<ParamListener>(get_node());params_=param_listener_->get_params();
  if(!check_param_vector_size()||!params_.calibration_required||get_update_rate()!=520||params_.observation_history!=1||!params_.use_imu||params_.imu_sensor_name!="imu_sensor"||params_.gain_multiplier!=1||params_.estop_kd!=0||params_.init_duration!=0||params_.fade_in_duration!=0||params_.max_body_angle!=.6)
   throw std::runtime_error("Wheel lift requires reviewed 520 Hz position execution and calibration");
  nlohmann::json j;std::ifstream input(params_.model_path);input>>j;
  if(j.at("behavior")!="leg_lift_wheel_position"||j.at("contract")!=wheel_lift::Core::contract||j.at("in_shape")[1]!=27||j.at("observation_history")!=1||j.at("policy_action_size")!=8||j.at("motor_command_count")!=12||j.at("hub_observations")!=false||j.at("rotation_command_observed")!=false||j.at("checkpoint_sha256")!="44166928ee63d4d2e2f9e312d6cda26a864312457e72ed0bb10c5e83f8d2b740"||j.at("model_sha256")!=wheel_lift::WheelGeometry::model_sha256)
   throw std::runtime_error("Wrong approved wheel-blind policy or model contract");
  const std::array<std::string,4> legs{"front_r","front_l","back_r","back_l"};
  for(int i=0;i<12;++i){
   if(params_.joint_names[i]!="leg_"+legs[i/3]+"_"+std::to_string(i%3+1)||params_.action_types[i]!="position"||j.at("action_types")[i]!="position"||params_.kps[i]!=wheel_lift::Core::kp[i]||params_.kds[i]!=wheel_lift::Core::kd[i]||params_.init_kps[i]!=params_.kps[i]||params_.init_kds[i]!=params_.kds[i]||j.at("kps")[i]!=params_.kps[i]||j.at("kds")[i]!=params_.kds[i]||params_.default_joint_pos[i]!=wheel_lift::Core::nominal[i]||params_.action_scales[i]!=wheel_lift::Core::scale[i]||params_.joint_lower_limits[i]!=wheel_lift::Core::lo[i]||params_.joint_upper_limits[i]!=wheel_lift::Core::hi[i]||j.at("default_joint_pos")[i]!=params_.default_joint_pos[i]||j.at("action_scale")[i]!=params_.action_scales[i])throw std::runtime_error("Joint mapping, position modes, gains or limits differ");
  }
  std::ifstream network(params_.model_path);model_=RTNeural::json_parser::parseJson<float>(network,true);
  if(!model_||model_->layers.empty()||model_->getInSize()!=27||model_->getOutSize()!=8)throw std::runtime_error("Expected 27 inputs and 8 outputs");
  behavior_="leg_lift_wheel_position";policy_action_size_=8;single_observation_size_=27;
  fault_timer_=get_node()->create_wall_timer(std::chrono::milliseconds(100),[this]{int f=fault_.load();if(f&&f!=reported_fault_)RCLCPP_ERROR(get_node()->get_logger(),"Wheel lift stopped, fault %d: 1 clock, 2 sensors, 3 tilt, 4 model/PD, 5 stop, 6 activation",f);reported_fault_=f;});
  return controller_interface::CallbackReturn::SUCCESS;
 }catch(const std::exception& e){RCLCPP_ERROR(get_node()->get_logger(),"Wheel lift rejected: %s",e.what());return controller_interface::CallbackReturn::ERROR;}
}
void WheelLiftController::stop(int code){int expected=0;fault_.compare_exchange_strong(expected,code);estop_active_=true;for(auto& i:command_interfaces_)i.set_value(0.);}
bool WheelLiftController::sensors(wheel_lift::Sensors& s){
 for(int i=0;i<12;++i){auto& joint=state_interfaces_map_.at(params_.joint_names[i]);s.q[i]=joint.at("position").get().get_value();s.qd[i]=joint.at("velocity").get().get_value();}
 auto& imu=state_interfaces_map_.at("imu_sensor");const std::array<std::string,3> axes{"x","y","z"};
 for(int i=0;i<3;++i)s.angular[i]=imu.at("angular_velocity."+axes[i]).get().get_value();
 tf2::Quaternion quat(imu.at("orientation.x").get().get_value(),imu.at("orientation.y").get().get_value(),imu.at("orientation.z").get().get_value(),imu.at("orientation.w").get().get_value());
 if(!std::isfinite(quat.length2())||std::abs(quat.length2()-1)>.02)return false;
 quat.normalize();auto g=tf2::Matrix3x3(quat).transpose()*tf2::Vector3(0,0,-1);s.gravity={g.x(),g.y(),g.z()};
 // This hardware exposes IMU age but no individual encoder packet-age interface.
 imu_age_=imu.at("time_since_measurement_seconds").get().get_value();
 return wheel_lift::Core::finite(s)&&std::isfinite(imu_age_)&&imu_age_>=0&&imu_age_<=.1;
}
controller_interface::CallbackReturn WheelLiftController::on_activate(const rclcpp_lifecycle::State& state){
 try{
  auto result=NeuralController::on_activate(state);if(result!=controller_interface::CallbackReturn::SUCCESS){stop(6);return result;}
  for(const auto& name:params_.joint_names)for(const auto* field:{"position","velocity","effort","kp","kd"})command_interfaces_map_.at(name).at(field);
  wheel_lift::Sensors s;if(!sensors(s))throw std::runtime_error("Invalid/stale entry sensors");lift_.reset(s,startup_calibration_.wheel_home);lift_.press(); // Activation is the first button: pose.
  last_request_.reset();rt_leg_lift_command_ptr_.writeFromNonRT(nullptr);
  leg_lift_command_subscriber_=get_node()->create_subscription<std_msgs::msg::Int32>("/wheel_lift/advance",rclcpp::QoS(1).durability_volatile(),[this](std_msgs::msg::Int32::SharedPtr msg){rt_leg_lift_command_ptr_.writeFromNonRT(msg);});
  status_pub_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/lift_status",1);rt_status_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(status_pub_);rt_status_->msg_.data.resize(24);
  motor_pub_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/motor_commands",1);rt_motors_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(motor_pub_);rt_motors_->msg_.data.resize(72);
  rt_observation_publisher_->msg_.data.resize(27);rt_policy_output_publisher_->msg_.data.resize(8);rt_position_command_publisher_->msg_.data.resize(12);
  last_update_=-1;actor_clock_={};accepted_=rejected_=0;fault_=0;estop_active_=false;
  for(auto& i:command_interfaces_)i.set_value(0.);
  return controller_interface::CallbackReturn::SUCCESS;
 }catch(const std::exception& e){RCLCPP_ERROR(get_node()->get_logger(),"Lift activation rejected: %s",e.what());stop(6);return controller_interface::CallbackReturn::ERROR;}
}
controller_interface::CallbackReturn WheelLiftController::on_deactivate(const rclcpp_lifecycle::State& state){last_request_.reset();leg_lift_command_subscriber_.reset();return NeuralController::on_deactivate(state);}
controller_interface::CallbackReturn WheelLiftController::on_error(const rclcpp_lifecycle::State&){stop(4);return controller_interface::CallbackReturn::SUCCESS;}
controller_interface::return_type WheelLiftController::update(const rclcpp::Time& time,const rclcpp::Duration& period){
 const double now=time.seconds(),dt=period.seconds();const bool first=last_update_<0;
 try{
  if(estop_active_){stop(5);publish(dt);return controller_interface::return_type::OK;}
  if(!std::isfinite(now)||now<0||!std::isfinite(dt)||dt<0||dt>.01||(!first&&(dt==0||now<=last_update_||now-last_update_>.01))){stop(1);publish(dt);return controller_interface::return_type::OK;}
  last_update_=now;
  wheel_lift::Sensors s;if(!sensors(s)){stop(2);publish(dt);return controller_interface::return_type::OK;}
  if(-s.gravity[2]<std::cos(.6)){stop(3);publish(dt);return controller_interface::return_type::OK;}
  if(first&&dt==0){for(auto& i:command_interfaces_)i.set_value(0.);return controller_interface::return_type::OK;}
  lift_.measure(s);
  // Revalidate the verified angle before considering a lower press.
  lift_.tick(dt,s);
  auto ptr=rt_leg_lift_command_ptr_.readFromRT();
  if(ptr&&*ptr&&*ptr!=last_request_){last_request_=*ptr;if(last_request_->data!=1)throw std::invalid_argument("Expected advance event 1");if(lift_.press())++accepted_;else ++rejected_;}
  const bool inference=actor_clock_.tick(dt);
  if(inference){
   lift_.observe(s);model_->forward(lift_.observation.data());lift_.set_action(model_->getOutputs());
   if(rt_observation_publisher_->trylock()){std::copy(lift_.observation.begin(),lift_.observation.end(),rt_observation_publisher_->msg_.data.begin());rt_observation_publisher_->unlockAndPublish();}
   if(rt_policy_output_publisher_->trylock()){std::copy(lift_.action.begin(),lift_.action.end(),rt_policy_output_publisher_->msg_.data.begin());rt_policy_output_publisher_->unlockAndPublish();}
  }
  lift_.positions(s);
  for(int i=0;i<12;++i){auto& joint=command_interfaces_map_.at(params_.joint_names[i]);joint.at("position").get().set_value(lift_.command[i]);joint.at("velocity").get().set_value(0.);joint.at("effort").get().set_value(0.);joint.at("kp").get().set_value(params_.kps[i]);joint.at("kd").get().set_value(params_.kds[i]);}
  if(inference){if(rt_position_command_publisher_->trylock()){std::copy(lift_.command.begin(),lift_.command.end(),rt_position_command_publisher_->msg_.data.begin());rt_position_command_publisher_->unlockAndPublish();}publish(dt);}
 }catch(const std::exception&){stop(4);publish(dt);}
 return controller_interface::return_type::OK;
}
void WheelLiftController::publish(double dt){
 if(rt_status_&&rt_status_->trylock()){auto& d=rt_status_->msg_.data;
  d[0]=lift_.stage;d[1]=lift_.leg;d[2]=lift_.actor_command();d[3]=lift_.supported;d[4]=lift_.ready;d[5]=lift_.verified;d[6]=fault_.load();d[7]=accepted_;d[8]=rejected_;d[9]=dt;d[10]=imu_age_;d[11]=lift_.elapsed;d[12]=lift_.qualified;d[13]=lift_.settled;
  std::copy(lift_.margins.begin(),lift_.margins.end(),d.begin()+14);std::copy(lift_.reference.begin(),lift_.reference.end(),d.begin()+17);d[21]=lift_.velocity;d[22]=lift_.goal[lift_.leg];d[23]=lift_.rotating;rt_status_->unlockAndPublish();}
 if(rt_motors_&&rt_motors_->trylock()){auto& d=rt_motors_->msg_.data;const std::array<const char*,5> fields{"position","velocity","effort","kp","kd"};for(int i=0;i<12;++i){for(int j=0;j<5;++j)d[6*i+j]=command_interfaces_map_.at(params_.joint_names[i]).at(fields[j]).get().get_value();d[6*i+5]=lift_.estimated_pd[i];}rt_motors_->unlockAndPublish();}
}
}
PLUGINLIB_EXPORT_CLASS(neural_controller::WheelLiftController,controller_interface::ControllerInterface)
