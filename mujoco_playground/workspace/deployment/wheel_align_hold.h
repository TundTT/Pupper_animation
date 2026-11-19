#pragma once
#include <algorithm>
#include <array>
#include <cmath>

// Apply after the existing default + scale * action mapping, before writing
// wheel position commands. Position PD remains the motor's existing controller.
// Inputs/order: front_r, front_l, back_r, back_l; selected=-1 means stand.
// Call once per 20 ms inference tick, independently of the motor servo frequency.
// Preserve the calibrated goals and this latch across successive leg commands.
struct WheelAlignHold {
  std::array<bool, 4> held{};
  std::array<int, 4> settle_ticks{};

  void reset() { held.fill(false); settle_ticks.fill(0); }

  std::array<double, 4> apply(
      const std::array<double, 4>& position,
      const std::array<double, 4>& velocity,
      const std::array<double, 4>& calibrated_goal,
      const std::array<double, 4>& learned_target,
      int selected) {
    auto result = learned_target;
    for (int leg = 0; leg < 4; ++leg) {
      const double difference = calibrated_goal[leg] - position[leg];
      const double error = std::atan2(std::sin(difference), std::cos(difference));
      const bool settled = leg == selected && std::abs(error) < 0.08
                          && std::abs(velocity[leg]) < 0.25;
      settle_ticks[leg] = settled ? std::min(settle_ticks[leg] + 1, 5) : 0;
      held[leg] = held[leg] || settle_ticks[leg] >= 5;
      if (held[leg]) result[leg] = position[leg] + error;
    }
    return result;
  }
};
