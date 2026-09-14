#pragma once
#include "neural_controller/leg_to_wheel/geometry.hpp"
#include <stdexcept>

namespace leg_to_wheel {
using V8 = std::array<double, 8>;
using V12 = std::array<double, 12>;

inline constexpr std::array<int, 8> rows{0, 1, 3, 4, 6, 7, 9, 10};
// stand, FL, FR, BR, BL -> leg index (FR=0, FL=1, BR=2, BL=3). Same
// command/leg-index convention as keyframe_align and wheel_align_hybrid.
inline constexpr std::array<int, 5> legs{-1, 1, 0, 2, 3};

// SHIFT: reposition the still-leg-shaped support limbs (this direction is not
// alignment-constrained the way wheel-base align is, so the support stance is
// free to change). LIFT: raise the active leg clear of the ground, still in
// leg shape. HOLD_UNLOADED: hold there indefinitely while the operator heats
// and the shin passively recovers to its wheel shape -- this is
// operator/material-paced, not timed, so there is deliberately no auto-advance
// out of this phase. LOWER: contact-adaptive descent (see stall detection
// below) back onto the now-shorter, wheel-shaped limb. RECENTER: brief settle
// dwell to confirm the stall was a real touchdown before committing. HOLD:
// idle, ready for the next leg.
enum Phase { ENTRY, SHIFT, LIFT, HOLD_UNLOADED, LOWER, RECENTER, HOLD, STOPPED };

struct Config {
  double entry_seconds = 2.0, shift_seconds = 3.0, lift_seconds = 6.0,
         lower_seconds = 8.0, recenter_seconds = 2.0;

  // Per-joint (abduction, hip) speed/accel caps. Deliberately slower than
  // keyframe_align's wheel-align defaults (0.30-0.45 rad/s, 1.5-2 rad/s^2):
  // "slow and gentle" was an explicit requirement for this maneuver, and
  // unlike wheel-align there is no trained residual policy backstopping it.
  std::array<double, 2> speed_limit{0.15, 0.20};
  std::array<double, 2> accel_limit{0.6, 0.8};

  // Ground-contact detection for LOWER. There is no force/torque sensing on
  // this hardware -- robot_info/QUADMORPH.md: "Joint-state effort is a
  // command-derived estimate, not measured motor torque" -- so touchdown is
  // inferred from the active hip's measured-vs-commanded position tracking
  // error growing past normal PD lag, not a force threshold.
  double stall_error_rad = 0.05;
  double stall_settle_seconds = 0.3;
  // Conservative backstop in case stall detection is late or misses: the
  // descent ramp itself never commands the active hip past this magnitude.
  // This is a safety floor, not the primary touchdown signal.
  double lower_hip_limit_rad = 0.35;

  // Unweighted-lift and support-shift hip magnitudes (sign applied per leg,
  // matching the rest of this codebase's FR/BR positive, FL/BL negative
  // convention). PLACEHOLDER pending motion-capture, same practice as
  // wheel_align_reference_data.hpp's poses[] table -- default shift target
  // equals the current pose (a no-op), so an unconfigured build behaves like
  // "hold the support legs fixed" rather than silently applying an invented
  // stance. Fill in real per-stage values once measured/tuned.
  double lift_hip_rad = 1.2;
  bool shift_support_legs = false;
  double shift_support_hip_rad = 0.9;

  // Coarse clearance gates using leg_to_wheel::Geometry, analogous to
  // keyframe_align's rotation_floor_clearance_m/wheel/body gates, but shape-
  // aware via the `converted` bitmask. See geometry.hpp for what these are
  // and are not validated against.
  double lift_clearance_m = 0.015;
  double self_clearance_m = 0.010;

  void validate() const {
    for (double x : {entry_seconds, shift_seconds, lift_seconds, lower_seconds, recenter_seconds,
                      stall_error_rad, stall_settle_seconds, lower_hip_limit_rad, lift_hip_rad,
                      shift_support_hip_rad, lift_clearance_m, self_clearance_m})
      if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("Positive finite timing/limit required");
    for (double x : speed_limit) if (!std::isfinite(x) || x <= 0 || x > 1.0)
      throw std::invalid_argument("Speed limit out of hardware-plausible range");
    for (double x : accel_limit) if (!std::isfinite(x) || x <= 0 || x > 4.0)
      throw std::invalid_argument("Accel limit out of hardware-plausible range");
  }
};

struct Output {
  V8 position{};
  std::array<double, 4> hub_position{};
  int phase = STOPPED, active = 0, converted = 0;
  bool authority = false, blocked = false, stalled = false;
  double clearance = 0;
};

class Controller {
 public:
  Config config;
  Phase phase = STOPPED;
  int active = 0;     // command index (1..4) while a leg is up; 0 at HOLD
  int converted = 0;  // bitmask over leg index: already reshaped to wheel and planted
  Output output;

  static double smooth(double u) { u = std::clamp(u, 0., 1.); return u * u * u * (10 + u * (-15 + 6 * u)); }

  // `converted_at_entry` lets the caller carry state across activations --
  // this struct does not own that persistence, the same division of
  // responsibility as startup calibration elsewhere in this codebase
  // (wheel_align_hybrid.hpp: "Calibration is supplied by the startup owner,
  // never acquired on activation").
  void reset(const V12& q, int converted_at_entry) {
    config.validate();
    phase = ENTRY;
    active = 0;
    converted = converted_at_entry & 0xF;
    elapsed = stall_time = recenter_settled = 0;
    for (int a = 0; a < 8; ++a) applied[a] = start[a] = endpoint[a] = q[rows[a]];
    velocity.fill(0);
    for (int k = 0; k < 4; ++k) hub_hold[k] = q[3 * k + 2];
    output = Output{};
    output.converted = converted;
  }

  // requested: 0=hold, 1..4=FL,FR,BR,BL (matches `legs`/command_leg convention
  // used elsewhere). reformed_confirmed: operator signal that the lifted,
  // unloaded leg has recovered to its wheel shape -- the caller is
  // responsible for delivering this as a one-shot pulse, the same convention
  // already used for `requested` throughout this codebase (a ROS topic int,
  // not raw held-button state). stop: emergency/deactivation.
  Output step(double dt, int requested, bool reformed_confirmed, bool stop,
              const V12& q, const V12& qd, const std::array<double, 3>& gravity) {
    bool finite = std::isfinite(dt) && dt > 0 && dt <= .04 && requested >= 0 && requested <= 4;
    for (double x : q) finite = finite && std::isfinite(x);
    for (double x : qd) finite = finite && std::isfinite(x);
    for (double x : gravity) finite = finite && std::isfinite(x);
    if (stop || !finite) phase = STOPPED;
    if (phase == STOPPED) { output = Output{}; output.phase = STOPPED; output.converted = converted; return output; }

    if (phase == HOLD && requested != 0 && !((converted >> legs[requested]) & 1)) {
      active = requested;
      const int k = legs[active];
      endpoint = applied;
      if (config.shift_support_legs)
        for (int a = 0; a < 8; ++a)
          if (a / 2 != k) endpoint[a] = a % 2 == 1 ? sign_of(a / 2) * config.shift_support_hip_rad : applied[a];
      enter(SHIFT, endpoint);
    }

    const int k = std::max(legs[active], 0);
    const double sign = k % 2 == 0 ? 1. : -1.;
    elapsed += dt;
    double duration = 1;
    if (phase == ENTRY) duration = config.entry_seconds;
    if (phase == SHIFT) duration = config.shift_seconds;
    if (phase == LIFT) duration = config.lift_seconds;
    if (phase == LOWER) duration = config.lower_seconds;
    if (phase == RECENTER) duration = config.recenter_seconds;

    if (phase == SHIFT && elapsed / duration >= 1) {
      endpoint = applied;
      endpoint[2 * k + 1] = sign * config.lift_hip_rad;
      enter(LIFT, endpoint);
      duration = config.lift_seconds;
    }

    const bool ramping = phase != HOLD_UNLOADED && phase != HOLD;
    V8 desired = applied;
    if (ramping) for (int a = 0; a < 8; ++a) desired[a] = start[a] + smooth(elapsed / duration) * (endpoint[a] - start[a]);
    integrate(desired, dt);

    const double clearance = Geometry::clearance_below_support(q, gravity, k, converted);
    const bool self_ok = clearance >= (phase == LIFT || phase == HOLD_UNLOADED ? config.lift_clearance_m : config.self_clearance_m);

    bool stalled = false;
    if (phase == LOWER) {
      // Positive when the measured joint has fallen behind where the ramp
      // has commanded it to be, in the direction of travel (hip magnitude
      // decreasing toward the backstop): `sign` flips which raw sign of
      // (q - applied) that corresponds to for FR/BR vs FL/BL.
      const double lag = sign * (q[3 * k + 1] - applied[2 * k + 1]);
      stall_time = lag > config.stall_error_rad ? stall_time + dt : 0;
      if (stall_time >= config.stall_settle_seconds) {
        stalled = true;
        enter(RECENTER, applied);  // freeze exactly where contact was detected
      }
      // The backstop itself lives in the ramp target (see the LOWER entry
      // below), not as a post-hoc clamp here: clamping `applied` directly
      // after the velocity-limited integrator would produce an instantaneous
      // position jump on whichever step first reached the limit, defeating
      // the whole point of gentle motion.
    }

    double command_error = 0, speed = 0;
    for (int a = 0; a < 8; ++a) {
      command_error = std::max(command_error, std::abs(applied[a] - endpoint[a]));
      speed = std::max(speed, std::abs(velocity[a]));
    }
    const bool arrived = elapsed >= duration && command_error < .015 && speed < .04;

    if (phase == ENTRY && arrived) {
      phase = HOLD;
    } else if (phase == LIFT && arrived && self_ok) {
      for (int j = 0; j < 4; ++j) hub_hold[j] = q[3 * j + 2];
      phase = HOLD_UNLOADED;
    } else if (phase == HOLD_UNLOADED && reformed_confirmed) {
      endpoint = applied;
      // Ramp only down to the conservative backstop, not to the neutral/
      // standing hip value (0): touchdown is expected to arrive via stall
      // detection before this planned endpoint is reached. If the ramp
      // reaches the backstop with no stall ever detected, LOWER deliberately
      // holds there rather than completing -- that means real ground contact
      // was never observed, which needs a person or a higher-level fault
      // handler, not a silent "must have landed" assumption.
      endpoint[2 * k + 1] = sign * config.lower_hip_limit_rad;
      enter(LOWER, endpoint);
    } else if (phase == RECENTER) {
      recenter_settled = speed < .04 ? recenter_settled + dt : 0;
      if (recenter_settled >= config.recenter_seconds) {
        converted |= 1 << k;
        active = 0;
        phase = HOLD;
      }
    }

    output.position = applied;
    output.hub_position = hub_hold;
    output.phase = phase;
    output.active = active;
    output.converted = converted;
    output.authority = true;
    output.blocked = !self_ok;
    output.stalled = stalled;
    output.clearance = clearance;
    return output;
  }

 private:
  V8 applied{}, velocity{}, start{}, endpoint{};
  std::array<double, 4> hub_hold{};
  double elapsed = 0, stall_time = 0, recenter_settled = 0;

  static double sign_of(int leg) { return leg % 2 == 0 ? 1. : -1.; }

  void enter(Phase next, const V8& end) {
    phase = next;
    elapsed = 0;
    start = applied;
    endpoint = end;
    stall_time = 0;
    if (next == RECENTER) recenter_settled = 0;
  }

  void integrate(const V8& desired, double dt) {
    for (int a = 0; a < 8; ++a) {
      const int axis = a % 2;  // 0=abduction, 1=hip
      const double vmax = config.speed_limit[axis], accel = config.accel_limit[axis];
      const double dv = std::clamp(36 * (desired[a] - applied[a]) - 12 * velocity[a], -accel, accel) * dt;
      velocity[a] = std::clamp(velocity[a] + dv, -vmax, vmax);
      applied[a] += velocity[a] * dt;
    }
  }
};
}  // namespace leg_to_wheel
