#include "neural_controller/leg_to_wheel/controller.hpp"
#include <iostream>

using namespace leg_to_wheel;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

// Drives the controller against a simulated robot: physical joints (q, qd)
// track the controller's own commanded target exactly, UNLESS a simulated
// floor blocks a leg's hip during LOWER, in which case q freezes there and
// qd goes to zero -- there is no force sensor on the real hardware either
// (robot_info/QUADMORPH.md), so this is the same kind of signal the real
// deployment would have to work with. `previous_command` tracks the
// controller's own last output independently of q, because once a floor
// blocks q, q and the controller's still-ramping internal target legitimately
// diverge (that divergence *is* the stall signal) -- so gentleness must be
// checked against the controller's own command history, not against q.
struct Sim {
  Controller& c;
  V12 q, qd;
  V8 previous_command;
  double floor_hip_magnitude;  // 0 disables the floor

  Sim(Controller& controller, V12 initial_q, double floor)
      : c(controller), q(initial_q), qd{}, floor_hip_magnitude(floor) {
    for (int a = 0; a < 8; ++a) previous_command[a] = q[rows[a]];
  }

  Output advance(double dt, int request, bool confirmed, bool stop = false) {
    const auto o = c.step(dt, request, confirmed, stop, q, qd, {0, 0, -1});
    if (o.authority) {
      for (int a = 0; a < 8; ++a) {
        const int axis = a % 2;
        const double rate = (o.position[a] - previous_command[a]) / dt;
        require(std::abs(rate) <= c.config.speed_limit[axis] + 1e-6,
                "Controller's own commanded target exceeds its configured speed limit");
        previous_command[a] = o.position[a];
      }
    }
    for (int a = 0; a < 8; ++a) { qd[rows[a]] = (o.position[a] - q[rows[a]]) / dt; q[rows[a]] = o.position[a]; }
    for (int j = 0; j < 4; ++j) { qd[3 * j + 2] = (o.hub_position[j] - q[3 * j + 2]) / dt; q[3 * j + 2] = o.hub_position[j]; }
    if (o.phase == LOWER && floor_hip_magnitude > 0) {
      const int k = std::max(legs[o.active], 0);
      const double sign = k % 2 == 0 ? 1. : -1.;
      // Hip magnitude decreases from lift_hip_rad (1.2, clear of the ground)
      // toward 0 (home/standing) while lowering, so a physically-plausible
      // floor contact must sit strictly above lower_hip_limit_rad -- otherwise
      // the conservative blind ramp stops on its own before ever reaching a
      // (deeper-than-assumed) floor, and stall detection never gets exercised.
      if (sign * q[3 * k + 1] < floor_hip_magnitude) { q[3 * k + 1] = sign * floor_hip_magnitude; qd[3 * k + 1] = 0; }
    }
    return o;
  }
};

int main() {
  try {
    constexpr double dt = 1. / 520;
    const V12 start{1, 0, 3.13, -1, 0, -3.13, 1, 0, .4, -1, 0, -.4};

    // 1) Full four-leg cycle against a simulated floor: gentleness bounds
    // throughout, no-auto-advance out of HOLD_UNLOADED, progressive
    // `converted` bitmask, and a completed leg cannot be re-requested.
    {
      Controller c;
      c.reset(start, /*converted_at_entry=*/0);
      Sim sim(c, start, /*floor_hip_magnitude=*/0.5);

      for (int n = 0; n < 2000; ++n) sim.advance(dt, 0, false);

      for (int command = 1; command <= 4; ++command) {
        const int k = legs[command];
        int n = 0;
        bool saw_hold_unloaded = false, confirmed_sent = false;
        for (; n < 200000; ++n) {
          const bool at_hold_unloaded = c.phase == HOLD_UNLOADED;
          if (at_hold_unloaded) saw_hold_unloaded = true;
          // Prove HOLD_UNLOADED does not auto-advance: sit there for a while
          // without confirming, and require the phase to still be HOLD_UNLOADED.
          if (at_hold_unloaded && !confirmed_sent) {
            for (int wait = 0; wait < 500; ++wait) {
              const auto held = sim.advance(dt, command, false);
              require(held.phase == HOLD_UNLOADED, "HOLD_UNLOADED must not auto-advance without confirmation");
            }
            confirmed_sent = true;
          }
          sim.advance(dt, command, at_hold_unloaded && confirmed_sent);
          if (c.converted & (1 << k)) break;
        }
        require(saw_hold_unloaded, "Must pass through HOLD_UNLOADED on the way up");
        require(c.converted & (1 << k), "Leg must finish converted after a full cycle");
        require(n < 200000, "Cycle must complete within the test budget");
      }
      require(c.converted == 0xF, "All four legs converted after a full pass");

      // A completed leg cannot be re-requested (HOLD phase, all converted).
      const auto after = sim.advance(dt, 1, false);
      require(after.phase == HOLD, "Fully-converted leg request must be ignored");
    }

    // 2) Simulated ground contact well short of the conservative backstop:
    // the controller must find it via stall detection (not by riding the
    // planned ramp all the way to lower_hip_limit_rad, which would mean
    // treating an untouched, invented endpoint as "landed").
    {
      Controller c;
      c.config.lower_hip_limit_rad = 0.35;
      c.reset(start, 0);
      Sim sim(c, start, /*floor_hip_magnitude=*/0.5);
      const int command = 2;  // FR
      const int k = legs[command];
      const double sign = k % 2 == 0 ? 1. : -1.;

      bool confirmed_once = false, stalled_seen = false;
      double min_hip_seen = 999;
      for (int n = 0; n < 200000; ++n) {
        const bool confirm = c.phase == HOLD_UNLOADED;
        const auto o = sim.advance(dt, command, confirm);
        if (confirm) confirmed_once = true;
        if (o.phase == LOWER)
          require(sign * o.position[2 * k + 1] >= c.config.lower_hip_limit_rad - 1e-9,
                  "Descent must never command past the configured backstop");
        if (o.stalled) stalled_seen = true;
        if (o.phase == LOWER || o.phase == RECENTER) min_hip_seen = std::min(min_hip_seen, sign * o.position[2 * k + 1]);
        if (c.converted & (1 << k)) break;
      }
      require(confirmed_once, "Must have reached HOLD_UNLOADED and confirmed");
      require(stalled_seen, "Stall detector must fire when the ground blocks the commanded descent");
      require(c.converted & (1 << k), "Leg must still finish converted after a compliant landing");
      // Bound the plausible overshoot past the true contact point: the ramp
      // keeps advancing for the full stall_settle_seconds dwell before RECENTER
      // freezes it, so worst case is roughly
      // stall_error_rad + stall_settle_seconds * hip speed limit.
      const double max_overshoot = c.config.stall_error_rad + c.config.stall_settle_seconds * c.config.speed_limit[1] + 1e-6;
      require(min_hip_seen >= 0.5 - max_overshoot, "Stall must freeze near the true contact point, not overshoot it");
      require(min_hip_seen > c.config.lower_hip_limit_rad, "Real contact must be caught well before the conservative backstop");
    }

    // 3) Stop removes authority.
    {
      Controller c;
      V12 q{}, qd{};
      c.reset(q, 0);
      const auto stopped = c.step(dt, 0, false, true, q, qd, {0, 0, -1});
      require(!stopped.authority, "Stop must remove authority");
      require(stopped.phase == STOPPED, "Stop must report STOPPED");
    }

    std::cout << "leg_to_wheel_controller_test: all checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "leg_to_wheel_controller_test FAILED: " << e.what() << "\n";
    return 1;
  }
}
