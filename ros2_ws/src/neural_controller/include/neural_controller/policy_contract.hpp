#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <json.hpp>

namespace neural_controller {

// Optional export metadata. Legacy policies without it keep their old command
// behavior; walking exports declare the envelope used during training.
struct PolicyContract {
  bool bounded_commands = false;
  bool fixed_orientation = false;
  std::array<double, 3> low{}, high{}, orientation{};

  explicit PolicyContract(const nlohmann::json &j = nlohmann::json::object()) {
    if (j.contains("command_low") != j.contains("command_high")) {
      throw std::runtime_error("Policy must declare both command_low and command_high");
    }
    bounded_commands = j.contains("command_low");
    if (bounded_commands) {
      low = triple(j.at("command_low"));
      high = triple(j.at("command_high"));
      for (int i = 0; i < 3; ++i) {
        if (low[i] > 0.0 || high[i] < 0.0 || low[i] > high[i]) {
          throw std::runtime_error("Policy command bounds must contain zero");
        }
      }
    }
    fixed_orientation = j.contains("orientation_command");
    if (fixed_orientation) {
      orientation = triple(j.at("orientation_command"));
      double norm2 = 0.0;
      for (double value : orientation) norm2 += value * value;
      if (std::abs(norm2 - 1.0) > 1e-6) {
        throw std::runtime_error("Policy orientation_command must be a unit vector");
      }
    }
  }

  std::array<double, 3> command(std::array<double, 3> value) const {
    for (double x : value) {
      if (!std::isfinite(x)) return {0.0, 0.0, 0.0};
    }
    if (bounded_commands) {
      for (int i = 0; i < 3; ++i) value[i] = std::clamp(value[i], low[i], high[i]);
    }
    return value;
  }

 private:
  static std::array<double, 3> triple(const nlohmann::json &value) {
    if (!value.is_array() || value.size() != 3) {
      throw std::runtime_error("Policy command metadata must have three entries");
    }
    std::array<double, 3> result{};
    for (int i = 0; i < 3; ++i) {
      result[i] = value.at(i).get<double>();
      if (!std::isfinite(result[i])) throw std::runtime_error("Nonfinite policy metadata");
    }
    return result;
  }
};

inline void seed_observation_history(std::vector<float> &observation, int frame_size) {
  for (size_t offset = frame_size; offset < observation.size(); offset += frame_size) {
    std::copy_n(observation.begin(), frame_size, observation.begin() + offset);
  }
}

}  // namespace neural_controller
