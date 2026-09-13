#include "core.hpp"
#include <atomic>
#include <chrono>
#include <vector>
#include "controller_interface/controller_interface.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "robot_calibration/calibration.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace measurement_hold {
class CurrentPose:public controller_interface::ControllerInterface {
  Core core_;std::array<int,60> commands_{};std::array<int,24> states_{};
  std::atomic<int64_t> joy_time_{0};std::atomic<bool> stop_{true},ps_{true};
  std::atomic<double> gain_{0},error_{0};std::array<std::atomic<double>,12> target_{};
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr estop_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  static int64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  bool joy_ok()const {const auto age=now()-joy_time_.load();return joy_time_>0&&age>=0&&age<500000000&&!ps_;}
  void release(){for(auto&c:command_interfaces_)c.set_value(0.);gain_=0;}
  void read(Pose&q,Pose&v){for(int i=0;i<12;++i){q[i]=state_interfaces_.at(states_[2*i]).get_value();v[i]=state_interfaces_.at(states_[2*i+1]).get_value();}}
public:
  controller_interface::InterfaceConfiguration command_interface_configuration()const override {
    controller_interface::InterfaceConfiguration c{controller_interface::interface_configuration_type::INDIVIDUAL,{}};
    for(auto n:robot_calibration::joint_names)for(auto f:{"position","velocity","effort","kp","kd"})c.names.push_back(std::string(n)+"/"+f);return c;
  }
  controller_interface::InterfaceConfiguration state_interface_configuration()const override {
    controller_interface::InterfaceConfiguration c{controller_interface::interface_configuration_type::INDIVIDUAL,{}};
    for(auto n:robot_calibration::joint_names)for(auto f:{"position","velocity"})c.names.push_back(std::string(n)+"/"+f);return c;
  }
  controller_interface::CallbackReturn on_init()override {
    joy_=get_node()->create_subscription<sensor_msgs::msg::Joy>("/joy",rclcpp::QoS(1),[this](sensor_msgs::msg::Joy::SharedPtr msg){
      if(msg->buttons.size()<=10){joy_time_=0;stop_=true;return;}
      ps_=msg->buttons[10]!=0;if(ps_)stop_=true;joy_time_=now();
    });
    estop_=get_node()->create_subscription<std_msgs::msg::Empty>("/emergency_stop",10,[this](std_msgs::msg::Empty::SharedPtr){stop_=true;});
    publisher_=get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/status",10);
    timer_=get_node()->create_wall_timer(std::chrono::milliseconds(100),[this]{
      std_msgs::msg::Float64MultiArray m;m.data={stop_?1.:0.,gain_.load(),error_.load(),joy_ok()?1.:0.};
      for(auto&x:target_)m.data.push_back(x.load());publisher_->publish(m);
    });
    return controller_interface::CallbackReturn::SUCCESS;
  }
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State&)override{return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&)override {
    release();stop_=true;
    try {
      robot_calibration::CaptureLock lock;
      const auto session=robot_calibration::current_session();
      auto record=robot_calibration::read_json(robot_calibration::directory()/"measurement-hold.json");
      if(record.get<std::string>("encoder_session_id")!=session||!record.get<bool>("operator_confirmed")||
         record.get<std::string>("scope")!="current_position_measurement_only")throw std::runtime_error("Current-session measurement authorization required");
      auto reference=robot_calibration::numbers<12>(record,"positions");
      if(!joy_ok())throw std::runtime_error("Fresh gamepad with PS released required");
      auto cn=command_interface_configuration().names;auto sn=state_interface_configuration().names;
      for(size_t k=0;k<cn.size();++k){commands_[k]=-1;for(size_t j=0;j<command_interfaces_.size();++j)if(command_interfaces_[j].get_name()==cn[k])commands_[k]=j;if(commands_[k]<0)throw std::runtime_error("Missing command interface");}
      for(size_t k=0;k<sn.size();++k){states_[k]=-1;for(size_t j=0;j<state_interfaces_.size();++j)if(state_interfaces_[j].get_name()==sn[k])states_[k]=j;if(states_[k]<0)throw std::runtime_error("Missing feedback interface");}
      Pose q{},v{};read(q,v);
      for(int i=0;i<12;++i)if(std::abs(q[i]-reference[i])>.03)throw std::runtime_error("Pose changed since supported capture");
      core_.reset(q,v);for(int i=0;i<12;++i)target_[i]=q[i];
      stop_=false;
      RCLCPP_WARN(get_node()->get_logger(),"MEASUREMENT HOLD: current angles, 3 s stiffness ramp; PS/disconnect releases torque. NOT walking calibration.");
      return controller_interface::CallbackReturn::SUCCESS;
    }catch(const std::exception&e){RCLCPP_ERROR(get_node()->get_logger(),"Hold rejected: %s",e.what());release();return controller_interface::CallbackReturn::ERROR;}
  }
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&)override {stop_=true;core_.stop();release();return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&)override {stop_=true;core_.stop();release();return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::return_type update(const rclcpp::Time&,const rclcpp::Duration&dt)override {
    Pose q{},v{};read(q,v);
    if(stop_)core_.stop();
    if(dt.seconds()==0&&core_.elapsed==0){release();return controller_interface::return_type::OK;}
    core_.step(dt.seconds(),q,v,joy_ok());gain_=core_.gain;error_=core_.error;
    if(core_.stopped){stop_=true;release();return controller_interface::return_type::OK;}
    for(int i=0;i<12;++i){
      const double values[]{core_.target[i],0.,0.,7.5*core_.gain,.35*core_.gain};
      for(int j=0;j<5;++j)command_interfaces_[commands_[5*i+j]].set_value(values[j]);
    }
    return controller_interface::return_type::OK;
  }
};
}
PLUGINLIB_EXPORT_CLASS(measurement_hold::CurrentPose,controller_interface::ControllerInterface)
