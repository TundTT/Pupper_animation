#include "neural_controller/notebook_lift_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <fstream>
namespace neural_controller {
using Targets=notebook_alignment_v4::PositionTargets;
controller_interface::CallbackReturn NotebookLiftController::on_init(){
 try{
  param_listener_=std::make_shared<ParamListener>(get_node());params_=param_listener_->get_params();
  if(!check_param_vector_size()||!params_.calibration_required||get_update_rate()!=520||params_.repeat_action!=10||params_.observation_history!=4||!params_.use_imu||params_.imu_sensor_name!="imu_sensor"||params_.gain_multiplier!=1||params_.estop_kd!=0||params_.init_duration!=0||params_.fade_in_duration!=0||params_.max_body_angle!=.6)
   throw std::runtime_error("Lift requires exact520/52Hz, four frames, IMU and candidate execution settings");
  nlohmann::json j;std::ifstream input(params_.model_path);input>>j;
  if(j.at("behavior")!="notebook_lift_v4"||j.at("runtime_contract")!="quadmorph-notebook-lift-runtime-v1"||j.at("contract_id")!=Targets::contract||j.at("trained_task")!="lift"||j.at("normalization")!="fused"||j.at("normalization_inside_model")!=true||j.at("policy_action_size")!=8||j.at("motor_command_count")!=12)
   throw std::runtime_error("Wrong notebook lift model/ABI");
  if(j.at("checkpoint_sha256")!="bf27a972ef20f4187e0210a5416e775772d69ef71b5a27601616a087642a6ad1"||j.at("source_hashes").at("model.xml")!=notebook_alignment_v4::WheelGeometry::model_sha256)
   throw std::runtime_error("Wrong reviewed checkpoint or geometry");
  const auto& schema=j.at("schema");
  if(schema.at("frame_size")!=72||schema.at("history")!=4||schema.at("observations")!=288||schema.at("actions")!=8||schema.at("action_scale")!=.75||schema.at("target_rate")!=.1||schema.at("target_acceleration")!=2||std::abs(schema.at("actor_dt").get<double>()-10./520)>1e-12||std::abs(schema.at("physics_dt").get<double>()-1./520)>1e-12)
   throw std::runtime_error("Observation/action/timing schema differs from runtime");
  const auto& setting=j.at("runtime_settings");
  if(setting.at("actor_repeat")!=10||setting.at("command_hz")!=520||setting.at("desired_clearance_m")!=.008||setting.at("lift_seconds")!=4||setting.at("lower_seconds")!=4||setting.at("attempt_timeout_seconds")!=48||setting.at("support_settle_seconds")!=.5||setting.at("support_bottom_spread_m")!=.006||setting.at("imu_max_age_seconds")!=.1||setting.at("max_command_dt")!=.01||setting.at("stop_on_estimated_pd_above_nm")!=3||setting.at("hub_alignment_rotation")!=false||setting.at("model_to_encoder_proximal_offset")!=0)
   throw std::runtime_error("Immutable candidate runtime settings differ");
  const std::array<std::string,4> legs{"front_r","front_l","back_r","back_l"};
  for(int i=0;i<12;++i){
   if(params_.joint_names[i]!="leg_"+legs[i/3]+"_"+std::to_string(i%3+1)||params_.action_types[i]!="position"||j.at("joint_names")[i]!=params_.joint_names[i]||j.at("action_types")[i]!="position"||std::abs(params_.kps[i]-Targets::kp[i])>1e-7||std::abs(params_.kds[i]-Targets::kd[i])>1e-7||j.at("kps")[i].get<double>()!=params_.kps[i]||j.at("kds")[i].get<double>()!=params_.kds[i]||params_.init_kps[i]!=params_.kps[i]||params_.init_kds[i]!=params_.kds[i])
    throw std::runtime_error("Joint mapping, position modes or per-joint gains differ");
   if(i%3!=2){int a=2*(i/3)+i%3;if(std::abs(params_.joint_lower_limits[i]-Targets::lower[a])>1e-6||std::abs(params_.joint_upper_limits[i]-Targets::upper[a])>1e-6||params_.default_joint_pos[i]!=Targets::nominal[a]||params_.action_scales[i]!=.75)throw std::runtime_error("Proximal limits/nominal/scale differ");}
   else if(params_.joint_lower_limits[i]>-100||params_.joint_upper_limits[i]<100)throw std::runtime_error("Continuous hub profile required");
  }
  // Parse the unchanged learned graph. No generic12-action loader or old residual ABI.
  std::ifstream network(params_.model_path);model_=RTNeural::json_parser::parseJson<float>(network,true);
  if(!model_||model_->layers.empty()||model_->getInSize()!=288||model_->getOutSize()!=8)throw std::runtime_error("Expected288 inputs and8 outputs");
  behavior_="notebook_lift_v4";policy_action_size_=8;single_observation_size_=72;
  fault_timer_=get_node()->create_wall_timer(std::chrono::milliseconds(100),[this]{int f=fault_.load();if(f&&f!=reported_fault_)RCLCPP_ERROR(get_node()->get_logger(),"Notebook lift stopped, fault%d:1 clock,2 sensors/IMU-age,3 tilt,4 model/core/torque,5 operator stop,6 activation. Reactivate only after diagnosis.",f);reported_fault_=f;});
  return controller_interface::CallbackReturn::SUCCESS;
 }catch(const std::exception& e){RCLCPP_ERROR(get_node()->get_logger(),"Notebook lift rejected: %s",e.what());return controller_interface::CallbackReturn::ERROR;}
}
void NotebookLiftController::stop(int code){int expected=0;fault_.compare_exchange_strong(expected,code);estop_active_=true;for(auto& i:command_interfaces_)i.set_value(0.);}
bool NotebookLiftController::sensors(notebook_alignment_v4::LiftSensors& s){
 for(int i=0;i<12;++i){auto& joint=state_interfaces_map_.at(params_.joint_names[i]);s.q[i]=joint.at("position").get().get_value();s.qd[i]=joint.at("velocity").get().get_value();}
 auto& imu=state_interfaces_map_.at("imu_sensor");const std::array<std::string,3> axes{"x","y","z"};
 for(int i=0;i<3;++i)s.angular[i]=imu.at("angular_velocity."+axes[i]).get().get_value();
 tf2::Quaternion quat(imu.at("orientation.x").get().get_value(),imu.at("orientation.y").get().get_value(),imu.at("orientation.z").get().get_value(),imu.at("orientation.w").get().get_value());
 if(!std::isfinite(quat.length2())||std::abs(quat.length2()-1)>.02)return false;
 quat.normalize();auto g=tf2::Matrix3x3(quat).transpose()*tf2::Vector3(0,0,-1);s.gravity={g.x(),g.y(),g.z()};
 // This hardware exposes IMU age but no individual encoder packet-age interface.
 imu_age_=imu.at("time_since_measurement_seconds").get().get_value();
 return notebook_alignment_v4::LiftCore::finite(s)&&std::isfinite(imu_age_)&&imu_age_>=0&&imu_age_<=.1;
}
controller_interface::CallbackReturn NotebookLiftController::on_activate(const rclcpp_lifecycle::State& state){
 try{
  auto result=NeuralController::on_activate(state);if(result!=controller_interface::CallbackReturn::SUCCESS){stop(6);return result;}
  for(const auto& name:params_.joint_names)for(const auto* field:{"position","velocity","effort","kp","kd"})command_interfaces_map_.at(name).at(field);
  notebook_alignment_v4::LiftSensors s;if(!sensors(s))throw std::runtime_error("Invalid/stale entry sensors");lift_.reset(s,startup_calibration_.wheel_home);
  last_request_.reset();rt_leg_lift_command_ptr_.writeFromNonRT(nullptr);
  leg_lift_command_subscriber_=get_node()->create_subscription<std_msgs::msg::Int32>("/notebook_lift_command_index",rclcpp::QoS(1).durability_volatile(),[this](std_msgs::msg::Int32::SharedPtr msg){rt_leg_lift_command_ptr_.writeFromNonRT(msg);});
  status_pub_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/lift_status",1);rt_status_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(status_pub_);rt_status_->msg_.data.resize(17);
  motor_pub_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/motor_commands",1);rt_motors_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(motor_pub_);rt_motors_->msg_.data.resize(72);
  rt_observation_publisher_->msg_.data.resize(288);rt_policy_output_publisher_->msg_.data.resize(8);rt_position_command_publisher_->msg_.data.resize(12);
  last_update_=-1;actor_elapsed_=0;command_count_=0;fault_=0;estop_active_=false;
  for(auto& i:command_interfaces_)i.set_value(0.);
  return controller_interface::CallbackReturn::SUCCESS;
 }catch(const std::exception& e){RCLCPP_ERROR(get_node()->get_logger(),"Lift activation rejected: %s",e.what());stop(6);return controller_interface::CallbackReturn::ERROR;}
}
controller_interface::CallbackReturn NotebookLiftController::on_deactivate(const rclcpp_lifecycle::State& state){last_request_.reset();return NeuralController::on_deactivate(state);}
controller_interface::CallbackReturn NotebookLiftController::on_error(const rclcpp_lifecycle::State&){stop(4);return controller_interface::CallbackReturn::SUCCESS;}
controller_interface::return_type NotebookLiftController::update(const rclcpp::Time& time,const rclcpp::Duration& period){
 const double now=time.seconds(),dt=period.seconds();const bool first=last_update_<0;
 try{
  if(estop_active_){stop(5);publish(dt);return controller_interface::return_type::OK;}
  if(!std::isfinite(now)||now<0||!std::isfinite(dt)||dt<0||dt>.01||(!first&&(dt==0||now<=last_update_||now-last_update_>.01))){stop(1);publish(dt);return controller_interface::return_type::OK;}
  last_update_=now;
  notebook_alignment_v4::LiftSensors s;if(!sensors(s)){stop(2);publish(dt);return controller_interface::return_type::OK;}
  if(-s.gravity[2]<std::cos(.6)){stop(3);publish(dt);return controller_interface::return_type::OK;}
  if(first&&dt==0){for(auto& i:command_interfaces_)i.set_value(0.);return controller_interface::return_type::OK;}
  actor_elapsed_+=dt;
  const bool inference=(command_count_++%10)==0;
  if(inference){
   int event=-1;auto ptr=rt_leg_lift_command_ptr_.readFromRT();if(ptr&&*ptr&&*ptr!=last_request_){last_request_=*ptr;event=last_request_->data;if(event<0||event>4)throw std::invalid_argument("Invalid external command");}
   lift_.prepare(actor_elapsed_,event,s);actor_elapsed_=0;
   model_->forward(lift_.history.values.data());Targets::V8 a{};std::copy_n(model_->getOutputs(),8,a.begin());lift_.targets.set_action(a);
   if(rt_observation_publisher_->trylock()){std::copy(lift_.history.values.begin(),lift_.history.values.end(),rt_observation_publisher_->msg_.data.begin());rt_observation_publisher_->unlockAndPublish();}
   if(rt_policy_output_publisher_->trylock()){std::copy(a.begin(),a.end(),rt_policy_output_publisher_->msg_.data.begin());rt_policy_output_publisher_->unlockAndPublish();}
  }
  lift_.execute(dt,s);
  for(int i=0;i<12;++i){auto& joint=command_interfaces_map_.at(params_.joint_names[i]);joint.at("position").get().set_value(lift_.command[i]);joint.at("velocity").get().set_value(0.);joint.at("effort").get().set_value(0.);joint.at("kp").get().set_value(params_.kps[i]);joint.at("kd").get().set_value(params_.kds[i]);}
  if(inference){if(rt_position_command_publisher_->trylock()){std::copy(lift_.command.begin(),lift_.command.end(),rt_position_command_publisher_->msg_.data.begin());rt_position_command_publisher_->unlockAndPublish();}publish(dt);}
 }catch(const std::exception&){stop(4);publish(dt);}
 return controller_interface::return_type::OK;
}
void NotebookLiftController::publish(double dt){
 if(rt_status_&&rt_status_->trylock()){auto& d=rt_status_->msg_.data;auto& r=lift_.request;d[0]=r.phase;d[1]=r.leg;d[2]=r.pending;d[3]=r.height;d[4]=r.rate;d[5]=r.elapsed;d[6]=r.attempt;d[7]=r.timed_out;d[8]=r.recovery_failed;d[9]=lift_.supported;d[10]=fault_.load();d[11]=dt;d[12]=imu_age_;std::copy(lift_.bottoms.begin(),lift_.bottoms.end(),d.begin()+13);rt_status_->unlockAndPublish();}
 if(rt_motors_&&rt_motors_->trylock()){auto& d=rt_motors_->msg_.data;const std::array<const char*,5> fields{"position","velocity","effort","kp","kd"};for(int i=0;i<12;++i){for(int j=0;j<5;++j)d[6*i+j]=command_interfaces_map_.at(params_.joint_names[i]).at(fields[j]).get().get_value();d[6*i+5]=lift_.estimated_pd[i];}rt_motors_->unlockAndPublish();}
}
}
PLUGINLIB_EXPORT_CLASS(neural_controller::NotebookLiftController,controller_interface::ControllerInterface)
