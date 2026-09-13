#include "neural_controller/hub_roll_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "neural_controller/triangle_roll_contract.hpp"

namespace neural_controller {
void HubRollController::receive_joy(const sensor_msgs::msg::Joy& msg) {
  // DualSense /dev/input/js0: PS is index 10, R3 is 12. Match joy_utils.
  if(msg.buttons.size()<=10) { joy_receipt_ns_=0; return; }
  joy_stop_held_=msg.buttons[10]!=0;
  joy_receipt_ns_=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if(joy_stop_held_) estop_active_=true;
}
bool HubRollController::joy_ready() const {
  const auto now=std::chrono::steady_clock::now().time_since_epoch();
  const auto age=std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()-joy_receipt_ns_.load();
  return joy_receipt_ns_>0 && age>=0 && age<500000000 && !joy_stop_held_;
}
controller_interface::CallbackReturn HubRollController::on_init() {
  try {
    param_listener_=std::make_shared<ParamListener>(get_node()); params_=param_listener_->get_params();
    RCLCPP_INFO(get_node()->get_logger(),"HUB ONLY: hold measured proximal angles; 2 s hold + 12 s half-turn; no support adaptation or walking");
    if(!check_param_vector_size() || get_update_rate()!=520 || params_.repeat_action!=1 ||
       !params_.use_imu || params_.gain_multiplier!=1 || params_.estop_kd!=0)
      throw std::runtime_error("Triangle controller requires 520 Hz, IMU, unit gains and zero stop gains");
    nlohmann::json j; std::ifstream f(params_.model_path); f>>j;
    plan_hash_=j.at("plan_sha256");
    if(j!=nlohmann::json::parse(triangle_roll::contract_json))
      throw std::runtime_error("Unreviewed roll contract; rebuild from pinned export");
    triangle_.initial=j.at("initial").get<inverted_triangle::Pose>();
    triangle_.kp=j.at("kp").get<inverted_triangle::Pose>();
    triangle_.kd=j.at("kd").get<inverted_triangle::Pose>();
    for(int i=0;i<12;++i) {
      if(params_.joint_names[i]!=robot_calibration::joint_names[i] || j.at("joint_names")[i]!=params_.joint_names[i] ||
         params_.action_types[i]!="position" || params_.kps[i]!=triangle_.kp[i] || params_.kds[i]!=triangle_.kd[i] ||
         params_.joint_lower_limits[i]!=inverted_triangle::low[i] || params_.joint_upper_limits[i]!=inverted_triangle::high[i])
        throw std::runtime_error("Joint order/mode/gains/limits mismatch");
    }
    triangle_.validate();
    behavior_="triangle_roll"; single_observation_size_=6; params_.observation_history=1;
    triangle_joy_=get_node()->create_subscription<sensor_msgs::msg::Joy>("/joy",rclcpp::QoS(1),
      [this](sensor_msgs::msg::Joy::SharedPtr msg) {
        receive_joy(*msg);
      });
    motor_commands_publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/motor_commands",1);
    rt_motor_commands_publisher_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(motor_commands_publisher_);
    rt_motor_commands_publisher_->msg_.data.resize(60);
    alignment_status_publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/status",1);
    rt_alignment_status_publisher_=std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(alignment_status_publisher_);
    rt_alignment_status_publisher_->msg_.data.resize(16);
    fault_timer_=get_node()->create_wall_timer(std::chrono::milliseconds(100),[this]() {
      int code=fault_code_.load();
      if(code && code!=reported_fault_)
        RCLCPP_ERROR(get_node()->get_logger(),"HUB DIAGNOSTIC FAULT %d; torque released. Support robot; lifecycle reset required. See TRIANGLE_ROLL_LAB.md.",code);
      reported_fault_=code;
    });
    return controller_interface::CallbackReturn::SUCCESS;
  } catch(const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),"Triangle configuration rejected: %s",e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
}

controller_interface::CallbackReturn HubRollController::on_activate(const rclcpp_lifecycle::State& s) {
  release_torque();
  try {
    if(NeuralController::on_activate(s)!=controller_interface::CallbackReturn::SUCCESS)
      throw std::runtime_error("Startup calibration rejected");
    // Prevent recapture between validating the mapping and completing activation.
    robot_calibration::CaptureLock lock;
    inverted_triangle::Pose mapping{},q{},qd{}; keyframe_align::V3 w{},g{};
    if(params_.calibration_required) {
      if(!joy_ready()) throw std::runtime_error("Fresh gamepad with released stop button required");
      auto live=robot_calibration::load_current();
      nlohmann::json j; std::ifstream f(robot_calibration::directory()/"hub-roll-reference.json"); f>>j;
      if(live.calibration_id!=startup_calibration_.calibration_id || j.at("schema_version")!=1 ||
         j.at("calibration_id")!=live.calibration_id || j.at("plan_sha256")!=plan_hash_ ||
         j.at("operator_confirmed_tips_up")!=true || j.at("purpose")!="supported_hub_only_v1" ||
         j.at("joint_names").get<std::vector<std::string>>()!=params_.joint_names ||
         j.at("wheel_home").get<std::array<double,4>>()!=live.wheel_home)
        throw std::runtime_error("Missing/stale hub-only reference; use scripts/capture_hub_roll_reference.py");
      auto captured=j.at("captured_q").get<inverted_triangle::Pose>();
      if(!read_sensors(q,qd,w,g))throw std::runtime_error("Cannot read current reference pose");
      for(int i=0;i<12;++i)if(!std::isfinite(captured[i]) || std::abs(captured[i]-q[i])>.03)
        throw std::runtime_error("Robot moved since confirmed tips-up reference; do not auto-approach");
    }
    for(const auto& name:params_.joint_names) for(const auto& field:{"position","velocity","effort","kp","kd"})
      command_interfaces_map_.at(name).at(field);
    if(!read_sensors(q,qd,w,g) || -g[2]<std::cos(inverted_triangle::tilt_limit))
      throw std::runtime_error("Invalid/stale activation sensors or excessive tilt");
    for(auto speed:qd) if(std::abs(speed)>.1) throw std::runtime_error("Robot must be stationary");
    triangle_.reset(q,mapping); last_update_seconds_=-1; status_count_=0; fault_code_=0;
    consumed_.reset(); rt_leg_lift_command_ptr_.writeFromNonRT(nullptr);
    leg_lift_command_subscriber_=get_node()->create_subscription<std_msgs::msg::Int32>(
      "/hub_roll/command",rclcpp::QoS(1).durability_volatile(),
      [this](std_msgs::msg::Int32::SharedPtr msg){rt_leg_lift_command_ptr_.writeFromNonRT(msg);});
    release_torque();
    return controller_interface::CallbackReturn::SUCCESS;
  } catch(const std::exception& e) {
    triangle_.state=inverted_triangle::FAULT; triangle_.fault=inverted_triangle::ACTIVATION;
    fault_code_=triangle_.fault; estop_active_=true; release_torque();
    RCLCPP_ERROR(get_node()->get_logger(),"Triangle activation rejected: %s",e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
}
controller_interface::CallbackReturn HubRollController::on_deactivate(const rclcpp_lifecycle::State& s) {
  triangle_.stop(inverted_triangle::OPERATOR); consumed_.reset();
  return NeuralController::on_deactivate(s);
}
controller_interface::CallbackReturn HubRollController::on_error(const rclcpp_lifecycle::State&) {
  triangle_.stop(inverted_triangle::ACTIVATION); fault_code_=triangle_.fault; estop_active_=true;
  release_torque(); return controller_interface::CallbackReturn::SUCCESS;
}
controller_interface::return_type HubRollController::update(const rclcpp::Time& time,const rclcpp::Duration& period) {
  inverted_triangle::Pose q{},qd{}; keyframe_align::V3 w{},g{};
  const double now=time.seconds(),dt=period.seconds(); const bool first=last_update_seconds_<0;
  last_period_=dt;
  if(!std::isfinite(now) || now<0 || (!first && (now<=last_update_seconds_ || now-last_update_seconds_>.01)))
    triangle_.stop(inverted_triangle::TIMING);
  last_update_seconds_=now;
  if(!read_sensors(q,qd,w,g)) triangle_.stop(inverted_triangle::SENSORS);
  if(estop_active_) triangle_.stop(inverted_triangle::OPERATOR);
  if(params_.calibration_required && !joy_ready()) triangle_.stop(inverted_triangle::OPERATOR);
  auto ptr=rt_leg_lift_command_ptr_.readFromRT(); auto msg=ptr ? *ptr:nullptr;
  const bool fresh=msg && msg!=consumed_; if(fresh) consumed_=msg;
  if(first && dt==0) {
    if(-g[2]<std::cos(inverted_triangle::tilt_limit)) triangle_.stop(inverted_triangle::TILT);
    release_torque();
  } else {
    triangle_.step(dt,q,qd,hub_roll::V3(g[0],g[1],g[2]),msg ? msg->data:0,fresh);
    if(triangle_.state==inverted_triangle::FAULT) release_torque();
    else for(int i=0;i<12;++i) {
      auto& joint=command_interfaces_map_.at(params_.joint_names[i]);
      joint.at("position").get().set_value(triangle_.command[i]+triangle_.offset[i]);
      joint.at("velocity").get().set_value(0.); joint.at("effort").get().set_value(0.);
      joint.at("kp").get().set_value(triangle_.gain*triangle_.kp[i]);
      joint.at("kd").get().set_value(triangle_.gain*triangle_.kd[i]);
    }
  }
  fault_code_=triangle_.fault; publish_triangle();
  return controller_interface::return_type::OK;
}
void HubRollController::publish_triangle() {
  if(++status_count_%26) return;
  if(rt_motor_commands_publisher_->trylock()) {
    const std::array<const char*,5> fields{"position","velocity","effort","kp","kd"};
    for(int i=0;i<12;++i) for(int j=0;j<5;++j)
      rt_motor_commands_publisher_->msg_.data[5*i+j]=command_interfaces_map_.at(params_.joint_names[i]).at(fields[j]).get().get_value();
    rt_motor_commands_publisher_->unlockAndPublish();
  }
  if(rt_alignment_status_publisher_->trylock()) {
    auto& d=rt_alignment_status_publisher_->msg_.data;
    d[0]=triangle_.state; d[1]=triangle_.completed; d[2]=triangle_.segment; d[3]=triangle_.elapsed;
    d[4]=triangle_.fault; d[5]=triangle_.max_error; d[6]=triangle_.max_torque;
    d[7]=last_period_; d[8]=last_imu_age_; d[9]=triangle_.settled([&]{inverted_triangle::Pose q{}; for(int i=0;i<12;++i)q[i]=state_interfaces_map_.at(params_.joint_names[i]).at("position").get().get_value();return q;}(), [&]{inverted_triangle::Pose v{}; for(int i=0;i<12;++i)v[i]=state_interfaces_map_.at(params_.joint_names[i]).at("velocity").get().get_value();return v;}());
    for(int i=0;i<4;++i)d[10+i]=triangle_.support.loads[i];
    d[14]=triangle_.support.elapsed;
    d[15]=1.;
    rt_alignment_status_publisher_->unlockAndPublish();
  }
}
}
PLUGINLIB_EXPORT_CLASS(neural_controller::HubRollController,controller_interface::ControllerInterface)
