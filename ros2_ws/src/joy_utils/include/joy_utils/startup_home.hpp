#pragma once
#include <array>
#include <cmath>

namespace joy_utils {
// A startup-only encoder snapshot. Physical ring placement is an operator setup
// requirement; stationary encoders cannot identify which ring touches the ground.
struct StartupHome {
  bool captured = false, movement_requested = false, sampling = false;
  double since = 0, last = 0;
  std::array<double, 4> anchor{}, home{};
  bool observe(const std::array<double, 4>& q, const std::array<double, 4>& v, double now) {
    if (captured || movement_requested || !std::isfinite(now)) return false;
    for (int k=0; k<4; ++k) {
      if (!std::isfinite(q[k]) || !std::isfinite(v[k]) || std::abs(v[k]) > .05) {
        sampling = false; return false;
      }
    }
    bool stable = sampling && now > last && now-last < .2;
    for (int k=0; k<4; ++k)
      stable = stable && std::abs(std::atan2(std::sin(q[k]-anchor[k]), std::cos(q[k]-anchor[k]))) < .005;
    if (!stable) { anchor=q; since=now; sampling=true; }
    last=now;
    if (now-since < .5) return false;
    for (int k=0; k<4; ++k) home[k]=std::atan2(std::sin(q[k]),std::cos(q[k]));
    captured=true;
    return true;
  }
};
}
