// Real plugin/network with synthetic interfaces and a temporary synthetic calibration.
// No hardware nodes, CAN/SPI devices or robot controller-manager are used.
#include "neural_controller/notebook_lift_controller.hpp"
#include "pluginlib/class_loader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using neural_controller::NotebookLiftController;
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
class Harness:public NotebookLiftController {
 public:
 auto& params(){return params_;}auto& core(){return lift_;}int fault(){return fault_;}
 void command(int x){auto m=std::make_shared<std_msgs::msg::Int32>();m->data=x;rt_leg_lift_command_ptr_.writeFromNonRT(m);}
 void estop(){estop_active_=true;}
};
int main(int argc,char** argv){rclcpp::init(argc,argv);int result=0;
 auto dir=std::filesystem::temp_directory_path()/("notebook-lift-test-"+robot_calibration::identity());setenv("QUADMORPH_CALIBRATION_DIR",dir.c_str(),1);
 try{
 require(argc==3,"Expected candidate YAML and JSON");rclcpp::NodeOptions options;options.arguments({"--ros-args","--params-file",argv[1],"-p",std::string("model_path:=")+argv[2]});
 {pluginlib::ClassLoader<controller_interface::ControllerInterface> loader("controller_interface","controller_interface::ControllerInterface");auto plugin=loader.createSharedInstance("neural_controller/NotebookLiftController");require(plugin->init("neural_controller_notebook_lift","",520,"",options)==controller_interface::return_type::OK,"installed plugin/real model initializes");}
 Harness c;require(c.init("neural_controller_notebook_lift","",520,"",options)==controller_interface::return_type::OK,"init");require(c.on_configure({})==controller_interface::CallbackReturn::SUCCESS,"configure");
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
 // Frozen stand encoders exercise actual inference/execution without claiming a
 // simulated balance result. Requested changes stay within the PD guard here.
 for(int i=0;i<300;++i)tick();require(c.fault()==0,"actual model idle updates");c.command(2);for(int i=0;i<20;++i)tick();require(c.core().request.leg==0&&c.core().request.phase==1,"actual command subscription buffer drivesFR");
 for(int i=0;i<12;++i){require(output[i][3]==c.params().kps[i]&&output[i][4]==c.params().kds[i],"perjoint gains consumed");require(output[i][1]==0&&output[i][2]==0,"position PD without outer velocity command");if(i%3==2)require(output[i][0]==home[i/3],"hubs hold captured encoder winding");}
 if(c.core().alignment_enabled){for(int k=0;k<4;++k)require(std::abs(c.core().goal[k]-home[k]-M_PI)<1e-12,"calibrated start+180 goal");c.command(5);for(int i=0;i<20;++i)tick();require(c.core().rotation_requested&&!c.core().rotation_enabled,"rotate queues behind closed clearance gate");for(int k=0;k<4;++k)require(output[3*k+2][0]==home[k],"closed gate holds hub");}
 c.estop();tick();zero();require(c.fault()==5,"operator stop latches");
 auto reenter=[&]{c.on_deactivate({});for(int i=0;i<12;++i){q[i]=(i%3==2)?home[i/3]+8*M_PI:(i%3==0?(i/3%2?-1:1):0);qd[i]=0;}imu={0,0,0,0,0,0,1,0};assign();require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"reentry preserves fresh holds after revolutions");now=0;c.update(rclcpp::Time(int64_t(0),RCL_ROS_TIME),rclcpp::Duration::from_seconds(0));tick();require(c.fault()==0,"fresh entry");};
 reenter();for(int k=0;k<4;++k)require(output[3*k+2][0]==q[3*k+2],"fresh winding not old hold");
 imu[7]=.2;tick();zero();require(c.fault()==2,"stale IMU stops");
 reenter();qd[0]=NAN;tick();zero();require(c.fault()==2,"nonfinite encoder stops");
 reenter();c.command(-1);for(int i=0;i<10;++i)tick();zero();require(c.fault()==4,"invalid command rejected");
 reenter();now+=.02;c.update(rclcpp::Time(int64_t(now*1e9),RCL_ROS_TIME),rclcpp::Duration::from_seconds(.02));zero();require(c.fault()==1,"command clock gap stops");
 reenter();qd[2]=4.;tick();zero();require(c.fault()==4,"estimated total PD limit stops");
 c.on_deactivate({});zero();std::cout<<"PASS actual plugin lifecycle, trained inference, held winding, gains, calibration and stop guards on fake interfaces\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
 std::filesystem::remove_all(dir);rclcpp::shutdown();return result;}
