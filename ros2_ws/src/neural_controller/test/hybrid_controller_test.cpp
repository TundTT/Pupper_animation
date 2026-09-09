// Exercise the actual ROS plugin lifecycle and update path with loaned interfaces.
#include "neural_controller/neural_controller.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double a, double b, const char *message) { require(std::abs(a - b) < 1e-5, message); }
class Harness : public neural_controller::NeuralController {
 public:
  const auto &obs() const { return observation_; }
  const auto &hybrid() const { return hybrid_; }
  auto &hybrid() { return hybrid_; }
  auto &params() { return params_; }
  void zero_time() { init_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME); }
  void command(int value) {
    auto msg = std::make_shared<std_msgs::msg::Int32>(); msg->data = value;
    rt_leg_lift_command_ptr_.writeFromNonRT(msg);
  }
  void stop() { estop_active_ = true; }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    require(argc == 3, "usage: hybrid_controller_test CONFIG_YAML POLICY_JSON");
    Harness controller;
    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "--params-file", argv[1], "-p", std::string("model_path:=") + argv[2]});
    require(controller.init("neural_controller_wheel_align_hybrid", "", 520, "", options) ==
        controller_interface::return_type::OK, "plugin initialization with actual config");
    require(controller.on_configure(rclcpp_lifecycle::State()) ==
        controller_interface::CallbackReturn::SUCCESS, "configure");
    std::array<double, 12> q{1, 0, .3, -1, 0, -.4, 1, 0, .5, -1, 0, -.6}, qd{};
    std::array<std::array<double, 5>, 12> commands{};
    std::vector<hardware_interface::CommandInterface> ci;
    std::vector<hardware_interface::StateInterface> si;
    ci.reserve(60); si.reserve(32);
    const std::array<std::string, 5> types{"position", "velocity", "effort", "kp", "kd"};
    for (int i = 0; i < 12; ++i) {
      const auto &name = controller.params().joint_names[i];
      for (int j = 0; j < 5; ++j) ci.emplace_back(name, types[j], &commands[i][j]);
      si.emplace_back(name, "position", &q[i]); si.emplace_back(name, "velocity", &qd[i]);
    }
    std::array<double, 7> imu{.01, -.02, .03, 0, 0, 0, 1};
    const std::array<std::string, 7> imu_types{"angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "orientation.x", "orientation.y", "orientation.z", "orientation.w"};
    for (int i = 0; i < 7; ++i) si.emplace_back("imu_sensor", imu_types[i], &imu[i]);
    auto activate = [&] {
      std::vector<hardware_interface::LoanedCommandInterface> lc;
      std::vector<hardware_interface::LoanedStateInterface> ls;
      for (auto &x : ci) lc.emplace_back(x);
      for (auto &x : si) ls.emplace_back(x);
      controller.assign_interfaces(std::move(lc), std::move(ls));
      require(controller.on_activate(rclcpp_lifecycle::State()) ==
          controller_interface::CallbackReturn::SUCCESS, "activate");
      controller.zero_time();
    };
    auto tick = [&](double seconds) {
      require(controller.update(rclcpp::Time(static_cast<int64_t>(seconds * 1e9), RCL_ROS_TIME),
          rclcpp::Duration::from_seconds(.002)) == controller_interface::return_type::OK, "update");
    };
    activate();
    for (int k = 0; k < 4; ++k) q[3*k+2] += .1;
    tick(.5);
    for (int k = 0; k < 4; ++k) {
      near(commands[3*k+2][1], -.2, "startup holds all wheel snapshots");
      near(commands[3*k+2][3], 0, "startup wheel kp zero");
      near(commands[3*k+2][4], .35, "startup wheel velocity servo gain");
    }
    for (int i = 0; i < 12; ++i) qd[i] = .01 * (i+1);
    controller.command(1);
    tick(2.0);
    auto obs = controller.obs();
    require(obs.size() == 51 && controller.hybrid().leg() == 1, "51 observations and command 1 selects FL");
    for (int i = 0; i < 3; ++i) near(obs[i], imu[i], "angular velocity frame");
    near(obs[5], -1, "gravity frame");
    for (int i = 0; i < 5; ++i) near(obs[6+i], i == 1 ? 1 : 0, "effective command one hot");
    for (int i = 0; i < 12; ++i) {
      near(obs[11+i], i % 3 == 2 ? neural_controller::WheelAlignHybrid::wrap(q[i]) :
          q[i] - controller.params().default_joint_pos[i], "joint observation order");
      near(obs[23+i], .1 * qd[i], "velocity observation order");
    }
    for (int k = 0; k < 4; ++k) {
      double err = neural_controller::WheelAlignHybrid::wrap(controller.hybrid().target[k] - q[3*k+2]);
      near(obs[43+k], std::sin(err), "target sine block");
      near(obs[47+k], std::cos(err), "target cosine block");
    }
    // Replay the first inference: its last-action block was zero before update.
    auto first_input = obs;
    for (int i = 35; i < 43; ++i) first_input[i] = 0;
    std::ifstream weights(argv[2]);
    auto network = RTNeural::json_parser::parseJson<float>(weights, false);
    network->forward(first_input.data());
    for (int a = 0; a < 8; ++a) {
      int row = neural_controller::WheelAlignHybrid::position_rows[a];
      double action = network->getOutputs()[a];
      near(obs[35+a], action, "last action stores raw network output");
      near(commands[row][0], std::clamp(controller.params().default_joint_pos[row] +
          controller.params().action_scales[row] * action, controller.params().joint_lower_limits[row],
          controller.params().joint_upper_limits[row]), "8 outputs map directly to position rows");
    }
    // Changed command interrupts rotation before wheel output; the new leg waits.
    controller.hybrid().phase = neural_controller::WheelAlignHybrid::ROTATE;
    controller.command(2);
    for (int n = 1; n <= 10; ++n) tick(2.0 + .002*n);
    require(controller.hybrid().phase == neural_controller::WheelAlignHybrid::LOWER &&
        controller.hybrid().active_command == 1, "runtime command interruption");
    near(controller.obs()[6], 1, "runtime lowering observes stand");
    near(commands[5][1], -.35*qd[5], "interruption holds current FL encoder");
    controller.stop(); tick(2.022);  // between policy ticks
    for (const auto &c : commands) {
      near(c[0], 0, "estop position"); near(c[1], 0, "estop velocity");
      near(c[3], 0, "estop kp"); near(c[4], 1, "estop damping");
    }
    require(controller.on_deactivate(rclcpp_lifecycle::State()) ==
        controller_interface::CallbackReturn::SUCCESS, "deactivate");
    activate();
    require(controller.hybrid().phase == neural_controller::WheelAlignHybrid::IDLE &&
        controller.hybrid().command == 0, "reactivate clears pending command and phase");
    for (int a = 35; a < 43; ++a) near(controller.obs()[a], 0, "reactivate clears last actions");
    for (int k = 0; k < 4; ++k) near(controller.hybrid().home[k],
        neural_controller::WheelAlignHybrid::wrap(q[3*k+2]), "reactivate captures fresh session home");
    controller.stop(); tick(.1);
    for (const auto &c : commands) near(c[4], 1, "estop works during startup");
    controller.on_deactivate(rclcpp_lifecycle::State());
    std::cout << "PASS: actual YAML/plugin lifecycle, encoder/IMU observations, direct actions, wheel holds, interruption, estop, reactivation\n";
  } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; result = 1; }
  rclcpp::shutdown();
  return result;
}
