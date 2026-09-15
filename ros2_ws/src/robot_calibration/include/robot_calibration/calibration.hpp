#pragma once

// File I/O belongs in lifecycle/startup code, never in the motor update loop.
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace robot_calibration {
namespace fs = std::filesystem;
using Tree = boost::property_tree::ptree;
inline constexpr std::array<const char *, 12> joint_names{
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
    "leg_back_l_1", "leg_back_l_2", "leg_back_l_3"};
inline double wrap(double x) { return std::atan2(std::sin(x), std::cos(x)); }
inline fs::path directory() {
  fs::path dir;
  if (const auto *p = std::getenv("QUADMORPH_CALIBRATION_DIR")) dir = p;
  else if (const auto *p = std::getenv("XDG_STATE_HOME")) dir = fs::path(p) / "quadmorph";
  else if (const auto *p = std::getenv("HOME")) dir = fs::path(p) / ".local/state/quadmorph";
  if (!dir.is_absolute()) throw std::runtime_error("Calibration directory must be absolute");
  return dir;
}
inline std::string read_line(const fs::path &path) {
  std::ifstream stream(path);
  std::string value;
  if (!std::getline(stream, value) || value.empty())
    throw std::runtime_error("Cannot read " + path.string());
  return value;
}
inline std::string boot_id() { return read_line("/proc/sys/kernel/random/boot_id"); }
inline std::string process_start(int pid) {
  const auto stat = read_line("/proc/" + std::to_string(pid) + "/stat");
  const auto close = stat.rfind(')');
  if (close == std::string::npos) throw std::runtime_error("Invalid process identity");
  std::istringstream fields(stat.substr(close + 2));
  std::string value;
  for (int field = 3; field <= 22; ++field)
    if (!(fields >> value)) throw std::runtime_error("Invalid process start time");
  return value;
}
inline Tree read_json(const fs::path &path) {
  Tree tree;
  boost::property_tree::read_json(path.string(), tree);
  return tree;
}
inline std::string identity() {
  std::random_device random;
  std::ostringstream out;
  for (int i = 0; i < 4; ++i) out << std::hex << std::setw(8) << std::setfill('0') << random();
  return out.str();
}
class CaptureLock {
 public:
  CaptureLock() {
    fs::create_directories(directory());
    fd_ = ::open((directory() / "capture.lock").c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) throw std::runtime_error("Cannot open calibration lock");
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      ::close(fd_); fd_ = -1;
      throw std::runtime_error("Calibration/startup is already in progress; try again after it finishes");
    }
  }
  ~CaptureLock() { if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); } }
  CaptureLock(const CaptureLock &) = delete;
  CaptureLock &operator=(const CaptureLock &) = delete;
 private:
  int fd_ = -1;
};
inline void atomic_json(const fs::path &path, const Tree &tree) {
  const auto tmp = path.string() + "." + identity() + ".tmp";
  try {
    boost::property_tree::write_json(tmp, tree);
    fs::rename(tmp, path);
  } catch (...) { std::error_code ec; fs::remove(tmp, ec); throw; }
}
inline std::string begin_session() {
  const auto id = identity();
  Tree tree;
  tree.put("schema_version", 1);
  tree.put("session_id", id);
  tree.put("boot_id", boot_id());
  tree.put("owner_pid", ::getpid());
  tree.put("owner_start_ticks", process_start(::getpid()));
  tree.put("ready", false);
  atomic_json(directory() / "encoder-session.json", tree);
  return id;
}
inline void finish_session(const std::string &id) {
  auto tree = read_json(directory() / "encoder-session.json");
  if (tree.get<std::string>("session_id") != id) throw std::runtime_error("Encoder session changed");
  tree.put("ready", true);
  atomic_json(directory() / "encoder-session.json", tree);
}
inline void invalidate_session(const std::string &id) noexcept {
  if (id.empty()) return;
  try {
    const auto path = directory() / "encoder-session.json";
    if (read_json(path).get<std::string>("session_id") == id) fs::remove(path);
  } catch (...) { /* Missing/unreadable records cannot authorize motion. */ }
}
inline std::string current_session() {
  const auto tree = read_json(directory() / "encoder-session.json");
  if (tree.get<int>("schema_version") != 1 || !tree.get<bool>("ready") ||
      tree.get<std::string>("boot_id") != boot_id() ||
      tree.get<std::string>("owner_start_ticks") != process_start(tree.get<int>("owner_pid")))
    throw std::runtime_error("Encoder session is not ready or belongs to a previous robot process");
  const auto id = tree.get<std::string>("session_id");
  if (id.empty()) throw std::runtime_error("Empty encoder session");
  return id;
}
template <std::size_t N> std::array<double, N> numbers(const Tree &tree, const std::string &key) {
  const auto &items = tree.get_child(key);
  if (items.size() != N) throw std::runtime_error("Wrong calibration vector size: " + key);
  std::array<double, N> result{};
  std::size_t i = 0;
  for (const auto &item : items) {
    if (!item.first.empty()) throw std::runtime_error("Expected calibration array: " + key);
    result[i] = item.second.get_value<double>();
    if (!std::isfinite(result[i++])) throw std::runtime_error("Non-finite calibration angle");
  }
  return result;
}
struct Calibration {
  std::string session_id, calibration_id;
  std::array<double, 12> reference_joint_positions{};
  std::array<double, 4> wheel_home{}, wheel_base_target{};
};
inline Calibration load_current() {
  const auto session = current_session();
  const auto tree = read_json(directory() / "calibration.json");
  if (tree.get<int>("schema_version") != 1 || tree.get<std::string>("encoder_session_id") != session ||
      !tree.get<bool>("operator_confirmed") || tree.get<std::string>("angle_units") != "radians" ||
      tree.get<std::string>("reference_convention") != "marked_point_ring_home")
    throw std::runtime_error("Missing, incompatible or stale startup calibration; run ros2 run robot_calibration calibrate capture");
  const auto &names = tree.get_child("joint_names");
  if (names.size() != joint_names.size()) throw std::runtime_error("Wrong calibration joint count");
  std::size_t i = 0;
  for (const auto &name : names)
    if (!name.first.empty() || name.second.get_value<std::string>() != joint_names[i++])
      throw std::runtime_error("Calibration joint order mismatch");
  Calibration result;
  result.session_id = session;
  result.calibration_id = tree.get<std::string>("calibration_id");
  if (result.calibration_id.empty()) throw std::runtime_error("Empty calibration identity");
  result.reference_joint_positions = numbers<12>(tree, "reference_joint_positions");
  result.wheel_home = numbers<4>(tree, "wheel_home");
  result.wheel_base_target = numbers<4>(tree, "wheel_base_target");
  for (int k = 0; k < 4; ++k) {
    if (std::abs(wrap(result.wheel_base_target[k] - result.wheel_home[k] - std::acos(-1.0))) > 1e-9)
      throw std::runtime_error("Calibration base target disagrees with home + pi");
  }
  if (current_session() != session) throw std::runtime_error("Encoder session changed while reading calibration");
  return result;
}
}  // namespace robot_calibration
