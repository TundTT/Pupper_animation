// Fake interfaces only; run with ROS_DOMAIN_ID=193 and ROS_LOCALHOST_ONLY=1.
#include "../src/joint_pose_controller.cpp"
#include <rclcpp/rclcpp.hpp>
#include <iostream>
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T> void array(robot_calibration::Tree&t,const char*k,const T&v){robot_calibration::Tree a;for(auto x:v){robot_calibration::Tree e;e.put("",x);a.push_back({"",e});}t.add_child(k,a);}
int main(int argc,char**argv){rclcpp::init(argc,argv);int result=0;
 auto dir=std::filesystem::temp_directory_path()/("joint-pose-fake-"+robot_calibration::identity());setenv("QUADMORPH_CALIBRATION_DIR",dir.c_str(),1);
 try{
  neural_controller::JointPoseController c;require(c.init("joint_pose_fake","",520,"",rclcpp::NodeOptions())==controller_interface::return_type::OK,"init");
  c.on_configure({});joint_pose::Pose q{1,0,0,-1,0,0,1,0,0,-1,0,0},v{},goal=q;goal[1]=-.1;
  std::array<std::array<double,5>,12> out{};std::array<double,5> imu{0,0,0,1,0};
  std::vector<hardware_interface::CommandInterface> ci;std::vector<hardware_interface::StateInterface> si;ci.reserve(60);si.reserve(29);
  for(int i=0;i<12;++i){int k=0;for(auto f:{"position","velocity","effort","kp","kd"})ci.emplace_back(robot_calibration::joint_names[i],f,&out[i][k++]);
   si.emplace_back(robot_calibration::joint_names[i],"position",&q[i]);si.emplace_back(robot_calibration::joint_names[i],"velocity",&v[i]);}
  int k=0;for(auto f:{"orientation.x","orientation.y","orientation.z","orientation.w","time_since_measurement_seconds"})si.emplace_back("imu_sensor",f,&imu[k++]);
  std::vector<hardware_interface::LoanedCommandInterface> lc;std::vector<hardware_interface::LoanedStateInterface> ls;
  for(auto&x:ci)lc.emplace_back(x);for(auto&x:si)ls.emplace_back(x);c.assign_interfaces(std::move(lc),std::move(ls));
  auto zero=[&]{for(auto a:out)for(auto x:a)require(x==0,"all60 must be zero");};
  require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"missing calibration rejected");zero();
  auto session=robot_calibration::begin_session();robot_calibration::finish_session(session);
  robot_calibration::Tree cal;cal.put("schema_version",1);cal.put("calibration_id","fixture");cal.put("encoder_session_id",session);cal.put("operator_confirmed",true);
  cal.put("angle_units","radians");cal.put("reference_convention","marked_point_ring_home");array(cal,"joint_names",robot_calibration::joint_names);array(cal,"reference_joint_positions",q);
  std::array<double,4> homes{},bases{};for(auto&x:bases)x=robot_calibration::wrap(M_PI);array(cal,"wheel_home",homes);array(cal,"wheel_base_target",bases);robot_calibration::atomic_json(dir/"calibration.json",cal);
  robot_calibration::Tree request;request.put("calibration_id","fixture");request.put("operator_confirmed_supported",true);request.put("scope","supported_joint_pose");array(request,"target_positions",goal);array(request,"captured_positions",q);robot_calibration::atomic_json(dir/"joint-pose-request.json",request);
  require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"missing joystick rejected");zero();
  auto node=rclcpp::Node::make_shared("fake_pad");auto pub=node->create_publisher<sensor_msgs::msg::Joy>("/joy",1);
  auto pad=[&](bool pressed){sensor_msgs::msg::Joy m;m.buttons.resize(13);m.buttons[10]=pressed;for(int n=0;n<10;++n){pub->publish(m);rclcpp::spin_some(c.get_node()->get_node_base_interface());rclcpp::spin_some(node);std::this_thread::sleep_for(std::chrono::milliseconds(10));}};
  pad(false);require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"valid activation");
  c.update(rclcpp::Time(0),rclcpp::Duration::from_seconds(0));
  for(int i=0;i<12;++i){require(out[i][0]==q[i],"initial reference matches feedback");for(int j=1;j<5;++j)require(out[i][j]==0,"initial gains and feedforward zero");}
  for(int n=0;n<100;++n)c.update(rclcpp::Time(0),rclcpp::Duration::from_seconds(1./520));
  require(out[0][3]>0&&out[2][3]>0,"gains routed");for(auto a:out)require(a[1]==0&&a[2]==0,"no feedforward");
  pad(true);c.update(rclcpp::Time(0),rclcpp::Duration::from_seconds(.002));zero();
  pad(false);c.update(rclcpp::Time(0),rclcpp::Duration::from_seconds(.002));zero();
  require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"explicit fresh activation");
  imu[4]=.2;c.update(rclcpp::Time(0),rclcpp::Duration::from_seconds(.002));zero();imu[4]=0;
  require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"reactivate for tilt");
  imu[0]=std::sin(.2);imu[3]=std::cos(.2);c.update(rclcpp::Time(0),rclcpp::Duration::from_seconds(.002));zero();
  c.on_deactivate({});zero();
  std::cout<<"PASS: actual ROS wrapper, calibration/gamepad gates, gains, PS latch, stale IMU, tilt, deactivation\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
 std::filesystem::remove_all(dir);rclcpp::shutdown();return result;
}
