// Real plugin/network with synthetic interfaces and a temporary synthetic calibration.
// No hardware nodes, CAN/SPI devices or robot controller-manager are used.
#include "neural_controller/leg_to_wheel_controller.hpp"
#include "pluginlib/class_loader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using neural_controller::LegToWheelController;
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
class Harness:public LegToWheelController {
 public:
 auto& params(){return params_;}auto& core(){return sequence_;}int fault(){return fault_;}
 void command(int x){auto m=std::make_shared<std_msgs::msg::Int32>();m->data=x;rt_leg_lift_command_ptr_.writeFromNonRT(m);}
 void estop(){estop_active_=true;}
};
int main(int argc,char** argv){rclcpp::init(argc,argv);int result=0;
 auto dir=std::filesystem::temp_directory_path()/("manual-conversion-test-"+robot_calibration::identity());setenv("QUADMORPH_CALIBRATION_DIR",dir.c_str(),1);
 try{
 require(argc==3,"Expected candidate YAML and JSON");rclcpp::NodeOptions options;options.arguments({"--ros-args","--params-file",argv[1],"-p",std::string("model_path:=")+argv[2]});
 {pluginlib::ClassLoader<controller_interface::ControllerInterface> loader("controller_interface","controller_interface::ControllerInterface");auto plugin=loader.createSharedInstance("neural_controller/LegToWheelController");require(plugin->init("neural_controller_leg_to_wheel","",520,"",options)==controller_interface::return_type::OK,"installed plugin/real model initializes");}
 Harness c;require(c.init("neural_controller_leg_to_wheel","",520,"",options)==controller_interface::return_type::OK,"init");require(c.on_configure({})==controller_interface::CallbackReturn::SUCCESS,"configure");
 std::array<double,12> q{1,0,.3,-1,0,-.4,1,0,.5,-1,0,-.6},qd{};std::array<std::array<double,5>,12> output{};
 std::vector<hardware_interface::CommandInterface> commands;std::vector<hardware_interface::StateInterface> states;commands.reserve(60);states.reserve(32);
 const std::array<std::string,5> fields{"position","velocity","effort","kp","kd"};
 for(int i=0;i<12;++i){for(int j=0;j<5;++j)commands.emplace_back(c.params().joint_names[i],fields[j],&output[i][j]);states.emplace_back(c.params().joint_names[i],"position",&q[i]);states.emplace_back(c.params().joint_names[i],"velocity",&qd[i]);}
 std::array<double,8> imu{0,0,0,0,0,0,1,0};const std::array<std::string,8> sensors{"angular_velocity.x","angular_velocity.y","angular_velocity.z","orientation.x","orientation.y","orientation.z","orientation.w","time_since_measurement_seconds"};for(int i=0;i<8;++i)states.emplace_back("imu_sensor",sensors[i],&imu[i]);
 auto assign=[&]{std::vector<hardware_interface::LoanedCommandInterface> ci;std::vector<hardware_interface::LoanedStateInterface> si;for(auto& x:commands)ci.emplace_back(x);for(auto& x:states)si.emplace_back(x);c.assign_interfaces(std::move(ci),std::move(si));};
 auto zero=[&]{for(auto& x:output)for(double v:x)require(v==0,"fault/stop commands zero");};
 assign();require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"missing calibration rejected");zero();
 auto session=robot_calibration::begin_session();robot_calibration::finish_session(session);std::array<double,4> home{q[2],q[5],q[8],q[11]},target{};for(int k=0;k<4;++k)target[k]=robot_calibration::wrap(home[k]+M_PI);
 nlohmann::json record={{"schema_version",1},{"calibration_id","synthetic-notebook-test"},{"encoder_session_id",session},{"operator_confirmed",true},{"angle_units","radians"},{"reference_convention","marked_point_ring_home"},{"joint_names",c.params().joint_names},{"reference_joint_positions",q},{"wheel_home",home},{"wheel_base_target",target}};{std::ofstream f(dir/"calibration.json");f<<record;}
 require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"synthetic current-session fixture accepted");
 c.update(rclcpp::Time(int64_t(0),RCL_ROS_TIME),rclcpp::Duration::from_seconds(0));zero();require(c.fault()==0,"first zero period accepted");
 double now=0;auto tick=[&]{now+=1./520;c.update(rclcpp::Time(int64_t(now*1e9),RCL_ROS_TIME),rclcpp::Duration::from_seconds(1./520));};
 // Frozen sensors check interface behavior, not physical lift or touchdown.
 for(int i=0;i<1100;++i)tick();require(c.fault()==0,"actual model updates without clearance/contact");
 require(c.core().step==1 && c.core().command()==1,"activation lifts FL and waits indefinitely");
 for(int i=0;i<12;++i){require(output[i][3]==5 && output[i][4]==.25,"export gains consumed");require(output[i][1]==0&&output[i][2]==0,"position commands only");}
 auto advance=[&](int step){c.command(step);for(int i=0;i<12;++i)tick();};
 advance(3);require(c.core().step==1,"out-of-order request cannot skip lower");
 for(int step=2;step<=8;++step){
   advance(step);require(c.core().step==step,"operator event accepted");
   require(c.core().command()==(step%2?(step+1)/2:0),"correct leg/stand command");
   advance(step);require(c.core().step==step,"duplicate request idempotent");
   for(int i=0;i<100;++i)tick();require(c.core().step==step && c.fault()==0,"no automatic progression");
 }
 advance(9);require(c.core().step==8,"sequence does not wrap");
 c.estop();tick();zero();require(c.fault()==5,"operator stop latches");
 advance(1);zero();require(c.core().step==8,"press cannot clear stop or restart");
 auto reenter=[&]{c.on_deactivate({});for(int i=0;i<12;++i){q[i]=(i%3==2)?home[i/3]+8*M_PI:(i%3==0?(i/3%2?-1:1):0);qd[i]=0;}imu={0,0,0,0,0,0,1,0};assign();require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"reentry preserves winding");now=0;c.update(rclcpp::Time(int64_t(0),RCL_ROS_TIME),rclcpp::Duration::from_seconds(0));for(int i=0;i<12;++i)tick();require(c.fault()==0,"fresh entry");};
 reenter();for(int k=0;k<4;++k)require(std::abs(output[3*k+2][0]-q[3*k+2])<=.8+1e-9,"target uses nearest turn, not old encoder winding");
 imu[7]=.2;tick();zero();require(c.fault()==2,"stale IMU stops");
 reenter();qd[0]=NAN;tick();zero();require(c.fault()==2,"nonfinite encoder stops");
 reenter();imu[3]=std::sin(.4);imu[6]=std::cos(.4);tick();zero();require(c.fault()==3,"tilt stops");
 reenter();now+=.02;c.update(rclcpp::Time(int64_t(now*1e9),RCL_ROS_TIME),rclcpp::Duration::from_seconds(.02));zero();require(c.fault()==1,"clock gap stops");
 c.on_deactivate({});zero();std::cout<<"PASS manual eight-step plugin, network, calibrated winding, gains, event idempotency and stop guards on fake interfaces\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
 std::filesystem::remove_all(dir);rclcpp::shutdown();return result;}
