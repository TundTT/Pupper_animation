// Fake interfaces only: exercise the installed plugin's actual lifecycle/update.
#include "neural_controller/keyframe_controller.hpp"
#include <filesystem>
#include <iostream>
#include "pluginlib/class_loader.hpp"

void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
class Harness:public neural_controller::KeyframeController {
 public:
  auto& params(){return params_;}auto& core(){return keyframes_;}
  bool network_loaded(){return bool(model_);}
  void command(int x){auto m=std::make_shared<std_msgs::msg::Int32>();m->data=x;rt_leg_lift_command_ptr_.writeFromNonRT(m);}
  void stop(){estop_active_=true;}
};
int main(int argc,char** argv){
  rclcpp::init(argc,argv);int result=0;
  const auto directory=std::filesystem::temp_directory_path()/("keyframe-plugin-fixture-"+robot_calibration::identity());
  setenv("QUADMORPH_CALIBRATION_DIR",directory.c_str(),1);
  try{
    require(argc==3,"Expected YAML and keyframe JSON");Harness c;rclcpp::NodeOptions options;
    options.arguments({"--ros-args","--params-file",argv[1],"-p",std::string("model_path:=")+argv[2]});
    {
      pluginlib::ClassLoader<controller_interface::ControllerInterface> loader("controller_interface","controller_interface::ControllerInterface");
      auto plugin=loader.createSharedInstance("neural_controller/KeyframeController");
      require(plugin->init("neural_controller_keyframe_align","",520,"",options)==controller_interface::return_type::OK,"Installed plugin discovery and initialization");
    }
    require(c.init("neural_controller_keyframe_align","",520,"",options)==controller_interface::return_type::OK,"init exact config");
    require(!c.network_loaded(),"No neural network in keyframe behavior");
    require(c.on_configure({})==controller_interface::CallbackReturn::SUCCESS,"configure");
    std::array<double,12> q{1,0,.3,-1,0,-.4,1,0,.5,-1,0,-.6},qd{};
    std::array<std::array<double,5>,12> outputs{};
    std::vector<hardware_interface::CommandInterface> commands;commands.reserve(60);
    std::vector<hardware_interface::StateInterface> states;states.reserve(32);
    const std::array<std::string,5> fields{"position","velocity","effort","kp","kd"};
    for(int i=0;i<12;++i){
      for(int j=0;j<5;++j)commands.emplace_back(c.params().joint_names[i],fields[j],&outputs[i][j]);
      states.emplace_back(c.params().joint_names[i],"position",&q[i]);states.emplace_back(c.params().joint_names[i],"velocity",&qd[i]);
    }
    std::array<double,8> imu{0,0,0,0,0,0,1,0};
    const std::array<std::string,8> sensors{"angular_velocity.x","angular_velocity.y","angular_velocity.z","orientation.x","orientation.y","orientation.z","orientation.w","time_since_measurement_seconds"};
    for(int i=0;i<8;++i)states.emplace_back("imu_sensor",sensors[i],&imu[i]);
    auto assign=[&]{std::vector<hardware_interface::LoanedCommandInterface> ci;std::vector<hardware_interface::LoanedStateInterface> si;
      for(auto& x:commands)ci.emplace_back(x);for(auto& x:states)si.emplace_back(x);c.assign_interfaces(std::move(ci),std::move(si));};
    assign();require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"Missing calibration rejects direct activation");
    const auto session=robot_calibration::begin_session();robot_calibration::finish_session(session);
    const std::array<double,4> home{q[2],q[5],q[8],q[11]};std::array<double,4> target{};
    for(int k=0;k<4;++k)target[k]=robot_calibration::wrap(home[k]+M_PI);
    nlohmann::json record={{"schema_version",1},{"calibration_id","test-fixture"},{"encoder_session_id",session},
      {"operator_confirmed",true},{"angle_units","radians"},{"reference_convention","marked_point_ring_home"},
      {"joint_names",c.params().joint_names},{"reference_joint_positions",q},{"wheel_home",home},{"wheel_base_target",target}};
    {std::ofstream f(directory/"calibration.json");f<<record;}
    require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"Confirmed fixture activates");
    double now=0;
    auto tick=[&]{now+=1./520;require(c.update(rclcpp::Time(int64_t(now*1e9),RCL_ROS_TIME),rclcpp::Duration::from_seconds(1./520))==controller_interface::return_type::OK,"update");
      for(int i=0;i<12;++i){const double previous=q[i];if(i%3==2)q[i]+=outputs[i][1]/520;else q[i]=outputs[i][0];qd[i]=(q[i]-previous)*520;}};
    for(int n=0;n<2200;++n)tick();
    for(int command=1;command<=4;++command){c.command(command);const int leg=keyframe_align::legs[command];
      for(int n=0;n<30000;++n){tick();if(c.core().completed&(1<<leg))break;}
      require(c.core().completed&(1<<leg),"Every mapped wheel completes via actual plugin");
      require(std::abs(robot_calibration::wrap(target[leg]-q[3*leg+2]))<.035,"Calibrated final target");
      for(int i=0;i<12;++i)require(outputs[i][3]==(i%3==2 ? 0.:5.),"Correct hardware gain routing");
    }
    auto check_brake=[&]{for(const auto& x:outputs)require(x[0]==0&&x[1]==0&&x[2]==0&&x[3]==0&&x[4]==1,"Stop disables position and velocity, keeps damping");};
    c.stop();tick();check_brake();c.on_deactivate({});
    for(int k=0;k<4;++k)q[3*k+2]=home[k]+16*M_PI+.2;qd.fill(0);
    for(int a=0;a<8;++a)q[keyframe_align::rows[a]]=keyframe_align::Geometry::neutral[a];
    assign();require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"Reentry shares live homes");
    require(c.core().completed==0&&c.core().active==0,"Reentry clears commands and completion");
    tick();for(int k=0;k<4;++k)require(std::abs(outputs[3*k+2][1])<1e-8,"Reentry holds current angles");
    imu[7]=.2;tick();check_brake();imu[7]=0;tick();check_brake();
    c.on_deactivate({});for(int a=0;a<8;++a)q[keyframe_align::rows[a]]=keyframe_align::Geometry::neutral[a];qd.fill(0);
    assign();require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"Explicit lifecycle reset clears stop");
    c.command(9);tick();check_brake();c.on_deactivate({});
    auto next=robot_calibration::begin_session();robot_calibration::finish_session(next);assign();
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"New encoder session invalidates fixture");
    std::cout<<"PASS: keyframe plugin, calibration, all-wheel mapping, stop, stale IMU, reentry, invalid commands\n";
  }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';result=1;}
  std::filesystem::remove_all(directory);rclcpp::shutdown();return result;
}
