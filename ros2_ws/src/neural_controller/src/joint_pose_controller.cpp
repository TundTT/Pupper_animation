#include "neural_controller/joint_pose.hpp"
#include <atomic>
#include <chrono>
#include "controller_interface/controller_interface.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "robot_calibration/calibration.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace neural_controller {
class JointPoseController:public controller_interface::ControllerInterface {
  joint_pose::Core core_;std::array<int,60> commands_{};std::array<int,29> states_{};
  std::atomic<int64_t> joy_time_{0};std::atomic<bool> stop_{true},ps_{true};
  std::atomic<int> state_{4},fault_{0};std::atomic<double> elapsed_{0},error_{0};
  std::array<std::atomic<double>,60> telemetry_{};
  bool allow_tilt_in_hold_=false;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr estop_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  static int64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  bool joy_ok()const {const auto age=now()-joy_time_.load();return joy_time_>0&&age>=0&&age<500000000&&!ps_;}
  void release(){for(auto&c:command_interfaces_)c.set_value(0.);for(auto&v:telemetry_)v=0.;}
  bool read(joint_pose::Pose&q,joint_pose::Pose&v,bool enforce_tilt=true) {
    for(int i=0;i<12;++i){q[i]=state_interfaces_.at(states_[2*i]).get_value();v[i]=state_interfaces_.at(states_[2*i+1]).get_value();}
    double x=state_interfaces_.at(states_[24]).get_value(),y=state_interfaces_.at(states_[25]).get_value();
    double z=state_interfaces_.at(states_[26]).get_value(),w=state_interfaces_.at(states_[27]).get_value();
    double age=state_interfaces_.at(states_[28]).get_value(),norm=x*x+y*y+z*z+w*w;
    return std::isfinite(norm)&&std::abs(norm-1)<.02&&std::isfinite(age)&&age>=0&&age<.1&&
      (!enforce_tilt || 1-2*(x*x+y*y)>std::cos(inverted_triangle::tilt_limit));
  }
 public:
  controller_interface::InterfaceConfiguration command_interface_configuration()const override {
    controller_interface::InterfaceConfiguration c{controller_interface::interface_configuration_type::INDIVIDUAL,{}};
    for(auto n:robot_calibration::joint_names)for(auto f:{"position","velocity","effort","kp","kd"})c.names.push_back(std::string(n)+"/"+f);return c;
  }
  controller_interface::InterfaceConfiguration state_interface_configuration()const override {
    controller_interface::InterfaceConfiguration c{controller_interface::interface_configuration_type::INDIVIDUAL,{}};
    for(auto n:robot_calibration::joint_names)for(auto f:{"position","velocity"})c.names.push_back(std::string(n)+"/"+f);
    for(auto f:{"orientation.x","orientation.y","orientation.z","orientation.w","time_since_measurement_seconds"})c.names.push_back(std::string("imu_sensor/")+f);return c;
  }
  controller_interface::CallbackReturn on_init()override {
    joy_=get_node()->create_subscription<sensor_msgs::msg::Joy>("/joy",rclcpp::QoS(1),[this](sensor_msgs::msg::Joy::SharedPtr m){
      if(m->buttons.size()<=10){joy_time_=0;stop_=true;return;}ps_=m->buttons[10]!=0;if(ps_)stop_=true;joy_time_=now();});
    estop_=get_node()->create_subscription<std_msgs::msg::Empty>("/emergency_stop",10,[this](std_msgs::msg::Empty::SharedPtr){stop_=true;});
    publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/status",10);
    timer_=get_node()->create_wall_timer(std::chrono::milliseconds(50),[this]{std_msgs::msg::Float64MultiArray m;
      m.data={double(state_),double(fault_),elapsed_.load(),error_.load(),joy_ok()?1.:0.};
      for(auto&v:telemetry_)m.data.push_back(v.load());publisher_->publish(m);});
    return controller_interface::CallbackReturn::SUCCESS;
  }
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State&)override{return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&)override {
    release();stop_=true;
    try {
      robot_calibration::CaptureLock lock;const auto cal=robot_calibration::load_current();
      auto request=robot_calibration::read_json(robot_calibration::directory()/"joint-pose-request.json");
      if(request.get<std::string>("calibration_id")!=cal.calibration_id||!request.get<bool>("operator_confirmed_supported")||
         request.get<std::string>("scope")!="supported_joint_pose")throw std::runtime_error("Session-bound supported request required");
      auto target=robot_calibration::numbers<12>(request,"target_positions");
      auto expected=robot_calibration::numbers<12>(request,"captured_positions");
      allow_tilt_in_hold_=request.get<bool>("allow_tilt_in_hold",false);
      if(!joy_ok())throw std::runtime_error("Fresh gamepad with PS released required");
      auto cn=command_interface_configuration().names;auto sn=state_interface_configuration().names;
      for(size_t k=0;k<cn.size();++k){commands_[k]=-1;for(size_t j=0;j<command_interfaces_.size();++j)if(command_interfaces_[j].get_name()==cn[k])commands_[k]=j;if(commands_[k]<0)throw std::runtime_error("Missing command");}
      for(size_t k=0;k<sn.size();++k){states_[k]=-1;for(size_t j=0;j<state_interfaces_.size();++j)if(state_interfaces_[j].get_name()==sn[k])states_[k]=j;if(states_[k]<0)throw std::runtime_error("Missing sensor");}
      joint_pose::Pose q{},v{};if(!read(q,v))throw std::runtime_error("Fresh level IMU required");
      for(int i=0;i<12;++i)if(std::abs(q[i]-expected[i])>.04)throw std::runtime_error("Pose changed since request");
      core_.reset(q,v,target);stop_=false;return controller_interface::CallbackReturn::SUCCESS;
    }catch(const std::exception&e){RCLCPP_ERROR(get_node()->get_logger(),"Joint pose rejected: %s",e.what());release();return controller_interface::CallbackReturn::ERROR;}
  }
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&)override {stop_=true;core_.stop(joint_pose::STOP);release();return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&)override {stop_=true;core_.stop(joint_pose::STOP);release();return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&dt)override {
    joint_pose::Pose q{},v{};if(stop_||!joy_ok())core_.stop(joint_pose::STOP);
    if(!read(q,v,!(allow_tilt_in_hold_&&core_.state==joint_pose::HOLD)))core_.stop(joint_pose::INPUT);
    if(!(dt.seconds()==0&&core_.total==0))core_.step(dt.seconds(),q,v);
    if(core_.state==joint_pose::FAULT){stop_=true;release();}
    else for(int i=0;i<12;++i){const double values[]{core_.reference[i],0.,0.,core_.kp[i],core_.kd[i]};
      for(int j=0;j<5;++j){command_interfaces_[commands_[5*i+j]].set_value(values[j]);telemetry_[5*i+j]=values[j];}}
    state_=core_.state;fault_=core_.fault;elapsed_=core_.total;error_=core_.error;
    return controller_interface::return_type::OK;
  }
};
}
PLUGINLIB_EXPORT_CLASS(neural_controller::JointPoseController,controller_interface::ControllerInterface)
