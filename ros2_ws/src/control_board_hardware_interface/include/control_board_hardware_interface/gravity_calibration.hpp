#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace gravity_calibration {
using Pose = std::array<double, 12>;

// Positions come from successful, checked SPI exchanges in the current activation.
// Velocity telemetry is not a stillness gate: the existing robot capture contract
// uses <=0.002 rad position excursion and operator confirmation.
class Sample {
 public:
  void reset() { count_ = 0; }
  bool observe(const Pose& q, double now, bool transport_valid) {
    if (!transport_valid || !std::isfinite(now) ||
        !std::all_of(q.begin(), q.end(), [](double x) { return std::isfinite(x); })) {
      reset(); return false;
    }
    if (count_ && (now <= last_ || now - last_ > .2)) reset();
    if (!count_) { first_ = now; low_ = high_ = q; }
    for (size_t i = 0; i < q.size(); ++i) {
      low_[i] = std::min(low_[i], q[i]); high_[i] = std::max(high_[i], q[i]);
      if (high_[i] - low_[i] > .002) { reset(); return false; }
    }
    last_ = now; measured_ = q; ++count_;
    return count_ >= 50 && now - first_ >= 1.;
  }
  Pose offsets(const Pose& reference) const {
    if (count_ < 50 || last_ - first_ < 1.) throw std::runtime_error("Gravity pose is not stationary");
    Pose result{};
    for (size_t i = 0; i < result.size(); ++i) {
      if (!std::isfinite(reference[i])) throw std::runtime_error("Nonfinite gravity reference");
      result[i] = measured_[i] - reference[i];
    }
    return result;
  }
 private:
  Pose low_{}, high_{}, measured_{};
  double first_ = 0., last_ = 0.;
  unsigned count_ = 0;
};
}  // namespace gravity_calibration
