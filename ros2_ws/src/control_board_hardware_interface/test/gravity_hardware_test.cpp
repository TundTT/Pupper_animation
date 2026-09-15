// Link the real hardware lifecycle against an in-memory SPI transport, never spidev.
#include "control_board_hardware_interface/control_board_hardware_interface.hpp"
#include <filesystem>
#include <limits>

static spi_command_t command{};
static spi_data_t feedback{};
static bool valid = true;
static unsigned exchanges = 0;
void init_spi() { command = {}; feedback = {}; }
spi_command_t* get_spi_command() { return &command; }
spi_data_t* get_spi_data() { return &feedback; }
bool spi_feedback_valid() { return valid; }
void require(bool pass, const char* what) { if (!pass) throw std::runtime_error(what); }
void spi_driver_run() {
  ++exchanges;
  for (int c = 0; c < 4; ++c) {
    // Any unexpected motion command during startup is a test failure.
    require(command.kp_abad[c] == 0 && command.kp_hip[c] == 0 && command.kp_knee[c] == 0,
            "startup commanded stiffness");
    require(command.kd_abad[c] == 0 && command.kd_hip[c] == 0 && command.kd_knee[c] == 0,
            "startup commanded damping");
    require(command.tau_abad_ff[c] == 0 && command.tau_hip_ff[c] == 0 && command.tau_knee_ff[c] == 0,
            "startup commanded effort");
    require(command.qd_des_abad[c] == 0 && command.qd_des_hip[c] == 0 && command.qd_des_knee[c] == 0,
            "startup commanded velocity");
    feedback.q_abad[c] = 6.f + c;
    feedback.q_hip[c] = -3.f - c;
    feedback.q_knee[c] = 15.f + c;  // Preserve arbitrary multi-turn hub readings.
    feedback.qd_abad[c] = feedback.qd_hip[c] = feedback.qd_knee[c] = 0;
  }
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  const auto dir = std::filesystem::temp_directory_path() / ("gravity-fake-" + robot_calibration::identity());
  setenv("QUADMORPH_CALIBRATION_DIR", dir.c_str(), 1);
  int result = 0;
  try {
    using control_board_hardware_interface::ControlBoardHardwareInterface;
    hardware_interface::HardwareInfo info;
    info.name = "fake_gravity_fixture"; info.type = "system";
    info.hardware_parameters["use_imu"] = "false";  // Never create an I2C device/thread.
    for (size_t i = 0; i < 12; ++i) {
      hardware_interface::ComponentInfo joint;
      joint.name = robot_calibration::joint_names[i]; joint.type = "joint";
      joint.parameters = {{"can_channel", std::to_string(i / 3 + 1)},
        {"can_id", std::to_string(i % 3 + 1)}, {"gravity_position", std::to_string(.1 * i)},
        {"position_min", "-1000"}, {"position_max", "1000"}, {"velocity_max", "30"},
        {"effort_max", "3"}, {"kp_max", "10"}, {"kd_max", "1"}};
      for (const char* name : {"position", "velocity", "effort", "kp", "kd"}) {
        hardware_interface::InterfaceInfo field; field.name = name;
        joint.command_interfaces.push_back(field);
      }
      for (const char* name : {"position", "velocity", "effort"}) {
        hardware_interface::InterfaceInfo field; field.name = name;
        joint.state_interfaces.push_back(field);
      }
      info.joints.push_back(joint);
    }
    ControlBoardHardwareInterface hw;
    using Return = hardware_interface::CallbackReturn;
    require(hw.on_init(info) == Return::SUCCESS, "init");
    require(hw.on_configure({}) == Return::SUCCESS, "configure");
    unsetenv("QUADMORPH_GRAVITY_CONFIRMED_BOOT");
    require(hw.on_activate({}) == Return::ERROR, "unconfirmed startup must fail");
    require(!std::filesystem::exists(dir / "encoder-session.json"), "unconfirmed session is not ready");
    // Explicit simulated confirmation, scoped to this fake transport process.
    setenv("QUADMORPH_GRAVITY_CONFIRMED_BOOT", robot_calibration::boot_id().c_str(), 1);
    require(hw.on_activate({}) == Return::SUCCESS, "stationary fake startup");
    require(exchanges >= 100, "must collect actual sample window");
    const auto session = robot_calibration::current_session();
    auto states = hw.export_state_interfaces();
    for (size_t i = 0; i < 12; ++i)
      require(std::abs(states[3*i].get_value() - .1*i) < 1e-5, "model reference mapping");
    require(std::filesystem::exists(dir / "gravity-offsets.json"), "offset audit");
    require(!std::filesystem::exists(dir / "calibration.json"), "startup must not self-confirm capture");
    require(hw.on_deactivate({}) == Return::SUCCESS, "deactivate");
    require(!std::filesystem::exists(dir / "encoder-session.json"), "deactivation invalidates session");
    require(hw.on_activate({}) == Return::ERROR, "cannot reuse confirmation on reactivation");
    setenv("QUADMORPH_GRAVITY_CONFIRMED_BOOT", robot_calibration::boot_id().c_str(), 1);
    require(hw.on_activate({}) == Return::SUCCESS, "fresh fake activation");
    require(robot_calibration::current_session() != session, "new session after zeroing");
    valid = false;
    require(hw.read(rclcpp::Time(0), rclcpp::Duration::from_seconds(.002)) == hardware_interface::return_type::ERROR,
            "transport loss must stop");
    require(!std::filesystem::exists(dir / "encoder-session.json"), "failed feedback invalidates session");
    for (auto flag : command.flags) require(flag == 0, "transport fault disables motors");
    std::cout << "PASS real lifecycle on fake SPI: fresh pose, zero gains, offsets, invalidation and confirmation\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; result = 1; }
  std::filesystem::remove_all(dir); rclcpp::shutdown(); return result;
}
