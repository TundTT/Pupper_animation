#include "neural_controller/inverted_triangle_controller.hpp"
#include <filesystem>
#include <iostream>
#include "pluginlib/class_loader.hpp"
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
class Harness:public neural_controller::InvertedTriangleController {
 public:
  auto& params(){return params_;} auto& core(){return triangle_;}
  void command(int n){auto m=std::make_shared<std_msgs::msg::Int32>();m->data=n;rt_leg_lift_command_ptr_.writeFromNonRT(m);}
  void joy(){joy_receipt_ns_=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  void stale_joy(){joy_receipt_ns_=1;}
  void stop(){estop_active_=true;}
};
int main(int argc,char** argv){
  rclcpp::init(argc,argv);int result=0;
  const auto dir=std::filesystem::temp_directory_path()/("triangle-fixture-"+robot_calibration::identity());
  setenv("QUADMORPH_CALIBRATION_DIR",dir.c_str(),1);
  try{
    require(argc==3,"Expected config YAML and plan JSON");Harness c;rclcpp::NodeOptions options;
    options.arguments({"--ros-args","--params-file",argv[1],"-p",std::string("model_path:=")+argv[2]});
    {pluginlib::ClassLoader<controller_interface::ControllerInterface> loader("controller_interface","controller_interface::ControllerInterface");
      auto plugin=loader.createSharedInstance("neural_controller/InvertedTriangleController");
      require(plugin->init("neural_controller_inverted_triangle","",520,"",options)==controller_interface::return_type::OK,"Installed plugin init");}
    require(c.init("neural_controller_inverted_triangle","",520,"",options)==controller_interface::return_type::OK,"Init");
    require(c.on_configure({})==controller_interface::CallbackReturn::SUCCESS,"Configure");
    auto q=c.core().initial;std::array<double,12> qd{};std::array<std::array<double,5>,12> output{};
    std::vector<hardware_interface::CommandInterface> commands;commands.reserve(60);
    std::vector<hardware_interface::StateInterface> states;states.reserve(32);
    for(int i=0;i<12;++i){int n=0;for(const auto& field:{"position","velocity","effort","kp","kd"})commands.emplace_back(c.params().joint_names[i],field,&output[i][n++]);
      states.emplace_back(c.params().joint_names[i],"position",&q[i]);states.emplace_back(c.params().joint_names[i],"velocity",&qd[i]);}
    std::array<double,8> imu{0,0,0,0,0,0,1,0};int n=0;
    for(const auto& field:{"angular_velocity.x","angular_velocity.y","angular_velocity.z","orientation.x","orientation.y","orientation.z","orientation.w","time_since_measurement_seconds"})states.emplace_back("imu_sensor",field,&imu[n++]);
    auto assign=[&]{std::vector<hardware_interface::LoanedCommandInterface> ci;std::vector<hardware_interface::LoanedStateInterface> si;
      for(auto& x:commands)ci.emplace_back(x);for(auto& x:states)si.emplace_back(x);c.assign_interfaces(std::move(ci),std::move(si));};
    auto zero=[&]{for(auto& a:output)for(auto x:a)require(x==0,"All command fields zero on rejection/stop");};
    assign();c.joy();require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"Missing calibration rejects");zero();
    auto session=robot_calibration::begin_session();robot_calibration::finish_session(session);
    std::array<double,4> home{.1,.2,.3,.4},base{};for(int k=0;k<4;++k)base[k]=robot_calibration::wrap(home[k]+M_PI);
    nlohmann::json record={{"schema_version",1},{"calibration_id","triangle-fixture"},{"encoder_session_id",session},{"operator_confirmed",true},
      {"angle_units","radians"},{"reference_convention","marked_point_ring_home"},{"joint_names",c.params().joint_names},
      {"reference_joint_positions",q},{"wheel_home",home},{"wheel_base_target",base}};
    {std::ofstream f(dir/"calibration.json");f<<record;}
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"Missing triangle map rejects even with startup calibration");zero();
    nlohmann::json plan;{std::ifstream f(argv[2]);f>>plan;}
    nlohmann::json map={{"schema_version",1},{"calibration_id","triangle-fixture"},{"plan_sha256",plan.at("plan_sha256")},
      {"operator_confirmed_inverted_start",true},{"axial_gap_m",.009},{"joint_names",c.params().joint_names},
      {"wheel_home",home},{"model_to_encoder_offset",std::array<double,12>{}},{"captured_q",q}};
    {std::ofstream f(dir/"inverted-triangle-map.json");f<<map;}
    c.stale_joy();require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"Missing gamepad rejects");zero();
    c.joy();require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"Confirmed fixtures activate");
    c.update(rclcpp::Time(int64_t(0),RCL_ROS_TIME),rclcpp::Duration::from_seconds(0));zero();
    double now=0;auto tick=[&]{c.joy();now+=1./520;c.update(rclcpp::Time(int64_t(now*1e9),RCL_ROS_TIME),rclcpp::Duration::from_seconds(1./520));};
    for(int i=0;i<1041;++i)tick();require(c.core().state==inverted_triangle::READY,"No automatic flip on activation");
    c.command(1);
    for(int n=0;n<25000 && !c.core().completed;++n){tick();
      require(c.core().state!=inverted_triangle::FAULT,"First leg tracks through real plugin");
      for(int i=0;i<12;++i){double next=output[i][0];qd[i]=(next-q[i])*520;q[i]=next;
        require(output[i][1]==0 && output[i][2]==0,"No feed-forward or velocity target");
        require(output[i][3]==(i%3==2?4.:5.) && output[i][4]==(i%3==2?.15:.25),"Simulation gains routed exactly");}}
    require(c.core().completed==1 && c.core().state==inverted_triangle::PAUSED,"Stops at rear-right planted checkpoint");
    imu[7]=.2;tick();zero();imu[7]=0;tick();zero();
    c.on_deactivate({});zero();q=c.core().initial;qd.fill(0);assign();c.joy();
    require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"Reactivation from correct initial pose");
    c.stale_joy();now+=.002;c.update(rclcpp::Time(int64_t(now*1e9),RCL_ROS_TIME),rclcpp::Duration::from_seconds(.002));zero();
    c.on_deactivate({});assign();c.joy();require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"Fresh lifecycle resets fault");
    c.stop();tick();zero();c.on_deactivate({});assign();
    auto next=robot_calibration::begin_session();robot_calibration::finish_session(next);
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"New encoder session invalidates old reference");zero();
    std::cout<<"PASS: installed triangle plugin, calibration/map/gamepad gates, rear-right motion, sensor/stop/reentry\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
  std::filesystem::remove_all(dir);rclcpp::shutdown();return result;
}
