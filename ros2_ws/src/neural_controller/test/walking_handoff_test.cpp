#include "neural_controller/neural_controller.hpp"
#include "neural_controller/triangle_roll_contract.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
class Harness:public neural_controller::NeuralController {
 public:
  auto& params(){return params_;}
  auto offsets(){return encoder_offset_;}
  auto observations(){return observation_;}
  auto origin(){return init_time_;}
  void stop(){estop_active_=true;}
};
int main(int argc,char**argv){
  rclcpp::init(argc,argv);int result=0;
  auto dir=std::filesystem::temp_directory_path()/("walking-handoff-"+robot_calibration::identity());
  std::filesystem::create_directories(dir);
  setenv("QUADMORPH_CALIBRATION_DIR",dir.c_str(),1);
  try {
    require(argc==3,"config and policy paths required");
    Harness c;rclcpp::NodeOptions options;
    options.arguments({"--ros-args","--params-file",argv[1],"-p",std::string("model_path:=")+argv[2],"-p","calibrated_walk_frame:=true"});
    require(c.init("neural_controller_walk_v2","",520,"",options)==controller_interface::return_type::OK,"walking init");
    require(c.on_configure({})==controller_interface::CallbackReturn::SUCCESS,"configure");
    std::array<double,12> q{},qd{};std::copy_n(c.params().default_joint_pos.begin(),12,q.begin());q[11]+=2*M_PI;
    std::array<std::array<double,5>,12> out{};
    std::vector<hardware_interface::CommandInterface> commands;
    std::vector<hardware_interface::StateInterface> states;
    commands.reserve(60);states.reserve(32);
    for(int i=0;i<12;++i){int n=0;for(auto f:{"position","velocity","effort","kp","kd"})commands.emplace_back(c.params().joint_names[i],f,&out[i][n++]);
      states.emplace_back(c.params().joint_names[i],"position",&q[i]);states.emplace_back(c.params().joint_names[i],"velocity",&qd[i]);}
    std::array<double,8> imu{0,0,0,0,0,0,1,0};int n=0;
    for(auto f:{"angular_velocity.x","angular_velocity.y","angular_velocity.z","orientation.x","orientation.y","orientation.z","orientation.w","time_since_measurement_seconds"})states.emplace_back("imu_sensor",f,&imu[n++]);
    auto assign=[&]{std::vector<hardware_interface::LoanedCommandInterface> ci;std::vector<hardware_interface::LoanedStateInterface> si;
      for(auto& x:commands)ci.emplace_back(x);for(auto& x:states)si.emplace_back(x);c.assign_interfaces(std::move(ci),std::move(si));};assign();
    auto session=robot_calibration::begin_session();robot_calibration::finish_session(session);
    std::array<double,4> homes{-1,1,-1,1},base{};for(int i=0;i<4;++i)base[i]=robot_calibration::wrap(homes[i]+M_PI);
    nlohmann::json record={{"schema_version",1},{"calibration_id","walk-test"},{"encoder_session_id",session},{"operator_confirmed",true},
      {"angle_units","radians"},{"reference_convention","marked_point_ring_home"},{"joint_names",c.params().joint_names},
      {"reference_joint_positions",q},{"wheel_home",homes},{"wheel_base_target",base}};
    {std::ofstream f(dir/"calibration.json");f<<record;}
    const auto plan=nlohmann::json::parse(triangle_roll::contract_json);
    nlohmann::json map={{"schema_version",2},{"calibration_id","walk-test"},{"plan_sha256",plan.at("plan_sha256")},
      {"operator_confirmed_inverted_start",true},{"axial_gap_m",.009},{"joint_names",c.params().joint_names},
      {"wheel_home",homes},{"model_to_encoder_offset",std::array<double,12>{}},{"captured_q",plan.at("initial")}};
    {std::ofstream f(dir/"triangle-roll-map.json");f<<map;}
    auto targets=q;targets[1]+=.08;std::array<double,12> kp{},kd{};kp.fill(4);kd.fill(.15);
    auto save_handoff=[&](double age){nlohmann::json j={{"calibration_id","walk-test"},{"joint_names",c.params().joint_names},
      {"source","completed_triangle_roll"},{"positions",targets},{"kp",kp},{"kd",kd},
      {"time_unix",std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()-age}};
      std::ofstream f(dir/"walking-handoff.json");f<<j;};save_handoff(0);
    require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"mapped activation");
    require(std::abs(c.offsets()[11]-2*M_PI)<1e-10,"full-turn offset");
    auto tick=[&](double t){c.update(c.origin()+rclcpp::Duration::from_seconds(t),rclcpp::Duration::from_seconds(1./520));};
    tick(0);
    for(int i=0;i<12;++i){require(std::abs(out[i][0]-targets[i])<1e-10,"first targets continuous");require(out[i][3]==4 && out[i][4]==.15,"first gains continuous");}
    tick(1);require(out[0][3]>4 && out[0][3]<5,"gains blend");
    tick(2.1);auto obs=c.observations();require(std::abs(obs[12+11])<1e-5,"actual model sees no full turn");
    require(out[11][0]>6 && out[11][0]<8.6,"actual motor output preserves full turn");
    c.stop();tick(2.2);require(out[11][3]==0,"stop removes position gain");
    c.on_deactivate({});assign();save_handoff(0);
    require(c.on_activate({})==controller_interface::CallbackReturn::SUCCESS,"reactivation");
    require(std::abs(c.offsets()[11]-2*M_PI)<1e-10,"no accumulated offset");
    c.on_deactivate({});assign();save_handoff(10);
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"stale snapshot rejects");
    std::filesystem::remove(dir/"walking-handoff.json");q[11]+=M_PI;
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"tips-up walking rejects");
    q[11]-=M_PI;map["calibration_id"]="old";
    {std::ofstream f(dir/"triangle-roll-map.json");f<<map;}
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"stale roll map rejects");
    auto newer=robot_calibration::begin_session();robot_calibration::finish_session(newer);
    require(c.on_activate({})==controller_interface::CallbackReturn::ERROR,"reboot session rejects");
    std::cout<<"Actual walking plugin: mapped observations/actions, continuous targets/gains, stop, reentry, stale reference passed\n";
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  std::filesystem::remove_all(dir);rclcpp::shutdown();return result;
}
