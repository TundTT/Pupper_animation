#pragma once
#include "wheel_align_hold.h"

// Wheel-only output replacement for the8-position/4-enable network variant.
// Initialize once at align activation, then preserve through all leg buttons.
struct WheelAlignSpin {
  WheelAlignHold hold;
  std::array<double, 4> reference{};
  std::array<int, 4> enable_ticks{};
  std::array<bool, 4> goal_commanded{};

  void reset(const std::array<double, 4>& encoder) {
    hold.reset(); reference = encoder; enable_ticks.fill(0); goal_commanded.fill(false);
  }

  std::array<double, 4> apply(
      const std::array<double, 4>& position,
      const std::array<double, 4>& velocity,
      const std::array<double, 4>& calibrated_goal,
      const std::array<double, 4>& network_enable,
      int selected) {
    hold.apply(position, velocity, calibrated_goal, reference, selected);
    auto target = reference;
    for (int leg = 0; leg < 4; ++leg) {
      const bool enable = leg == selected && network_enable[leg] > 0.;
      enable_ticks[leg] = enable ? std::min(enable_ticks[leg] + 1, 3) : 0;
      const bool turning = enable_ticks[leg] >= 3;
      if (leg == selected && !turning && !hold.held[leg] && !goal_commanded[leg])
        reference[leg] = position[leg];
      const double difference = calibrated_goal[leg] - reference[leg];
      const double error = std::atan2(std::sin(difference), std::cos(difference));
      if (turning) reference[leg] += std::clamp(error, -0.06, 0.06);
      const double remaining = calibrated_goal[leg] - reference[leg];
      goal_commanded[leg] = goal_commanded[leg] || (turning &&
          std::abs(std::atan2(std::sin(remaining), std::cos(remaining))) < 1e-4);
      if (hold.held[leg] || goal_commanded[leg]) reference[leg] = calibrated_goal[leg];
      const double delta = reference[leg] - position[leg];
      target[leg] = position[leg] + std::atan2(std::sin(delta), std::cos(delta));
    }
    return target;
  }
};
