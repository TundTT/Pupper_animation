#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace neural_controller {

// Direct port of attempt2_guard/source/hybrid_env.py. Joint rows are FR, FL, BR, BL;
// command indices are stand, FL, FR, BR, BL. No wheel-forward sign conversion.
struct WheelAlignHybrid {
  enum Phase { IDLE, LIFT, ROTATE, VERIFY, LOWER, HOLD };
  static constexpr std::array<int, 5> command_leg{-1, 1, 0, 2, 3};
  static constexpr std::array<int, 8> position_rows{0, 1, 3, 4, 6, 7, 9, 10};
  Phase phase = IDLE;
  int command = 0, active_command = 0;
  int phase_steps = 0, gate_steps = 0, settled_steps = 0;
  bool verified = false, was_rotating = false, step_pending = false;
  std::array<bool, 4> completed{};
  std::array<double, 4> home{}, target{}, hold{}, reference{};
  // Set once by the first-ever reset() (or an explicit recalibrate_home()) and never
  // cleared by reset() again -- see reset()'s comment for why.
  bool calibrated = false;

  static double wrap(double x) { return std::atan2(std::sin(x), std::cos(x)); }
  static double wheel_pd(double goal, double angle, double velocity) {
    return std::clamp(2.0 * wrap(goal - angle) - 0.35 * velocity, -2.0, 2.0);
  }
  bool up() const { return phase >= LIFT && phase <= VERIFY; }
  int leg() const { return std::max(command_leg[active_command], 0); }
  int effective_command() const { return up() ? active_command : 0; }

  // Called on every controller activation. Resets phase/command/gate state for a fresh
  // run, but deliberately does NOT recapture home/target/hold/reference once they've
  // been set once (see `calibrated`): the operator needs one ground-truth reference
  // established at robot-stack startup that survives switching to other controllers
  // (wheel/leg/etc.) and back -- re-homing on every single activation silently discards
  // that reference the moment any other policy has moved the wheels. Only the very first
  // activation after process start, or an explicit /wheel_align_hybrid_calibrate publish
  // while idle, updates the calibration -- see recalibrate_home().
  void reset(const std::array<double, 12> &q) {
    const bool was_calibrated = calibrated;
    const auto saved_home = home;
    const auto saved_target = target;
    const auto saved_hold = hold;
    const auto saved_reference = reference;
    *this = WheelAlignHybrid{};
    if (was_calibrated) {
      home = saved_home;
      target = saved_target;
      hold = saved_hold;
      reference = saved_reference;
      calibrated = true;
    } else {
      recalibrate_home(q);
    }
  }

  // Re-captures home/target/hold/reference from the current encoder angles without
  // touching phase/command state. Session-only by design: cross-boot zero repeatability
  // for these joints is unverified (the knee joints sharing this port's actuator/homing
  // path have shown up to ~33 deg of boot-to-boot drift), so persisting an absolute
  // angle to disk would silently go stale. Caller must only invoke this while
  // phase == IDLE (or HOLD with no leg mid-operation), so it never moves a live target.
  void recalibrate_home(const std::array<double, 12> &q) {
    for (int k = 0; k < 4; ++k) {
      home[k] = hold[k] = reference[k] = wrap(q[3 * k + 2]);
      target[k] = wrap(home[k] + std::acos(-1.0));
    }
    calibrated = true;
  }

  bool lift_ready(const std::array<double, 12> &q, double default_abduction,
                  const std::array<double, 3> &ang, double gravity_z) const {
    const int k = leg();
    const double signed_hip = q[3 * k + 1] * (k % 2 == 0 ? 1.0 : -1.0);
    const double norm = std::sqrt(ang[0] * ang[0] + ang[1] * ang[1] + ang[2] * ang[2]);
    return signed_hip > 1.1 && std::abs(q[3 * k] - default_abduction) < 0.35 &&
           -gravity_z > std::cos(0.12) && norm < 0.3;
  }

  void select_command(int requested, const std::array<double, 12> &q) {
    if (requested != command && up()) {
      hold[leg()] = wrap(q[3 * leg() + 2]);
      phase = LOWER;
      phase_steps = 0;
      verified = false;
    }
    command = requested;
    const int requested_leg = std::max(command_leg[command], 0);
    if ((phase == IDLE || phase == HOLD) && command != 0 && !completed[requested_leg]) {
      active_command = command;
      phase = LIFT;
      phase_steps = 0;
      reference = hold;
      verified = false;
    }
  }

  // Complete the preceding control interval using newly measured encoders. This
  // corresponds to the post-pipeline_step part of the environment, before the
  // next command selection/observation. No fabricated physics look-ahead.
  void finish_step(const std::array<double, 12> &q, const std::array<double, 12> &qd) {
    if (!step_pending) return;
    step_pending = false;
    const int k = leg(), wheel = 3 * k + 2, hip = 3 * k + 1;
    ++phase_steps;
    const double err = std::abs(wrap(target[k] - q[wheel]));
    const bool settled = was_rotating && err < 0.035 && std::abs(qd[wheel]) < 0.08;
    settled_steps = settled ? settled_steps + 1 : 0;
    Phase next = phase;
    if (phase == LIFT && gate_steps >= 10) next = ROTATE;
    else if (phase == ROTATE && err < 0.035) next = VERIFY;
    else if (phase == VERIFY && settled_steps >= 25) {
      next = LOWER;
      // Source latches at VERIFY -> LOWER, not after touchdown: actively hold
      // the settled encoder snapshot throughout lowering and subsequent HOLD.
      hold[k] = wrap(q[wheel]);
      verified = true;
    } else if (phase == LOWER && phase_steps >= 50 &&
               std::abs(q[hip]) < 0.25 && std::abs(qd[hip]) < 0.2) {
      next = HOLD;
      completed[k] = completed[k] || verified;
    }
    if (next != phase) { phase = next; phase_steps = 0; }
  }

  void begin_step(const std::array<double, 12> &q, bool gate, double dt) {
    const int k = leg();
    gate_steps = up() && gate ? gate_steps + 1 : 0;
    const bool rotating = (phase == ROTATE || phase == VERIFY) && gate;
    if (was_rotating && !rotating && up()) reference[k] = wrap(q[3 * k + 2]);
    was_rotating = rotating;
    if (rotating) reference[k] = wrap(reference[k] +
        std::clamp(wrap(target[k] - reference[k]), -0.25 * dt, 0.25 * dt));
    step_pending = true;
  }

  std::array<double, 4> wheel_commands(const std::array<double, 12> &q,
                                        const std::array<double, 12> &qd) const {
    std::array<double, 4> result{};
    for (int k = 0; k < 4; ++k) {
      const double goal = up() && k == leg() ? reference[k] : hold[k];
      result[k] = wheel_pd(goal, q[3 * k + 2], qd[3 * k + 2]);
    }
    return result;
  }
};
}  // namespace neural_controller
