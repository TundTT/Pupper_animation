#pragma once
#include <array>
#include <cmath>
#include <stdexcept>

namespace neural_controller {
// Fixed per-activation translation. Never wrap individual policy actions.
inline std::array<double,12> walking_offsets(
    const std::array<double,12>& q, const std::array<double,12>& home,
    const std::array<double,12>& reference, bool wheel_bridge = false) {
  std::array<double,12> offset{};
  constexpr double turn=6.2831853071795864769;
  for(int i=0;i<12;++i) {
    if(!std::isfinite(q[i]) || !std::isfinite(home[i]) || !std::isfinite(reference[i]))
      throw std::runtime_error("Nonfinite walking reference");
    if(i%3==2) offset[i]=reference[i]+turn*std::nearbyint((q[i]-home[i]-reference[i])/turn);
    else if(reference[i]!=0) throw std::runtime_error("Upper joints must retain their calibrated frame");
    // Only the verified zero-output bridge may enter from the wheel stance.
    // Hubs take the nearest home winding through its timed position ramp.
    const double tolerance = wheel_bridge && i%3==2 ? turn/2+1e-9 : .30;
    if(wheel_bridge && i%3==0) {
      const double sign=home[i]>0 ? 1. : -1.;
      if(sign*q[i]<.50 || sign*q[i]>std::abs(home[i])+.15)
        throw std::runtime_error("Bridge requires wheel or standing abduction pose");
    } else if(std::abs(q[i]-offset[i]-home[i])>tolerance)
      throw std::runtime_error("Walking requires a near-standing pose; finish roll first");
    if(i%3==2 && std::abs(home[i]+offset[i])+1.2>=1000.)
      throw std::runtime_error("Mapped walking envelope exceeds continuous hub limits");
  }
  return offset;
}
}
