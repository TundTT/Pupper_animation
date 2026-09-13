#pragma once
#include <array>

namespace neural_controller::policy_home {
// Shared wheel/walk export defaults, FR/FL/BR/BL, motor 1 then motor 2.
// Operator approved the corresponding live hold and floor stance 2026-09-13.
// This is a desired pose in the calibrated joint frame, never encoder zeroing.
inline constexpr std::array<double,8> upper{1,0,-1,0,1,0,-1,0};
constexpr std::array<double,12> with_hubs(const std::array<double,4>& hubs) {
  std::array<double,12> q{};
  for(int leg=0;leg<4;++leg) {
    q[3*leg]=upper[2*leg];q[3*leg+1]=upper[2*leg+1];q[3*leg+2]=hubs[leg];
  }
  return q;
}
}
