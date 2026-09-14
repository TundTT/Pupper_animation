#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stanford_stop_homing {
// Derived from do_homing() at 40116f60bd0f01506f5a4b22e95d2af0e4a81daa.
// Same advancing position target, PD torque estimate, alpha=.5 contact filter,
// and stop-angle offset. Bounds and confirmed return tracking are additions.
struct Joint {
  double start, position, target, velocity = 0, filtered = 0;
  double direction, model_stop, raw_stop = 0;
  bool homed = false;
  Joint(double raw, double dir, double angle)
      : start(raw), position(raw), target(raw), direction(dir), model_stop(angle) {}
  static constexpr double kp = 5.5, kd = .2, speed = 1.5, threshold = 2.;
  void observe(double q, double qd) {
    if (!std::isfinite(q) || !std::isfinite(qd)) throw std::runtime_error("Nonfinite homing feedback");
    if (std::abs(q-start)>4.1) throw std::runtime_error("Homing travel exceeded");
    if (std::abs(qd)>6.) throw std::runtime_error("Homing speed exceeded");
    position=q;
  }
  bool seek(double qd, double dt) {
    filtered = .5*filtered + .5*((target-position)*kp + (velocity-qd)*kd);
    if (std::abs(filtered)>=threshold) {
      raw_stop=position; target=position; velocity=0; homed=true; return true;
    }
    target+=direction*speed*dt; velocity=direction*speed;
    bound(qd);
    return false;
  }
  void bound(double qd) {
    // Bound the estimated TOTAL PD command, not just feed-forward effort.
    const double damping=(velocity-qd)*kd;
    target=std::clamp(target, position+(-2.5-damping)/kp,
                              position+(2.5-damping)/kp);
  }
  bool return_to_home(double qd, double dt, double home = 0.) {
    const double destination=raw_stop-model_stop+home;
    const double difference=destination-target;
    target+=std::clamp(difference,-speed*dt,speed*dt);
    velocity=std::abs(difference)>.01 ? std::copysign(speed,difference) : 0.;
    bound(qd);
    // Low-gain gravity-loaded joints need not settle to zero error. Stanford
    // only waited for the command ramp; additionally require measured proximity.
    return std::abs(position-destination)<.06 && std::abs(qd)<.2;
  }
  bool return_to_zero(double qd, double dt) { return return_to_home(qd,dt); }
};
}
