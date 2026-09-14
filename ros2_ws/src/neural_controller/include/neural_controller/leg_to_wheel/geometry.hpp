#pragma once
#include "neural_controller/wheel_align_geometry_data.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace leg_to_wheel {
using Vec = std::array<double, 3>;
using Mat = std::array<double, 9>;

// Reference distances for the two shin shapes this controller has to reason
// about. Neither number comes from a validated per-maneuver 3D collision
// sweep (compare flip's prepare_flip_geometry.py, in the `flip` worktree,
// which did that work for Phase 2's *rotating* flip and required a physical
// part redesign as a result). There is no rotation in leg-to-wheel, so that
// specific hazard does not apply, but these remain coarse vertical bounds,
// not a certified clearance -- confirm on the real robot before trusting them
// close to zero margin.
namespace shin_reference {
// robot_info/QUADMORPH.md, "Measured cold shapes" (measurements supplied
// 2026-09-13): hub-center-to-tip ranged 55-60mm across the 4 measured legs,
// cooled, tip-up. Midpoint used here; do not treat as exact for any one leg.
inline constexpr double leg_tip_hub_distance_m = 0.058;
// wheel_align_geometry_data.hpp / wheel_align_motion.hpp's nominal wheel
// radius, already used by the trained/hardware-tested align controllers.
inline constexpr double wheel_radius_m = 0.0505;
inline constexpr double wheel_half_width_m = 0.01675;
// Per the user's own reshaping notes: expect ~10mm length variation between
// individual appendages (manufacturing + thermal-cycle variance). Add this as
// a safety margin on top of the nominal figures above rather than trusting
// them exactly.
inline constexpr double appendage_length_uncertainty_m = 0.010;
}  // namespace shin_reference

// Per-leg ground-clearance estimate for a *mixed* stance -- some legs already
// reshaped to wheels (`converted` bit set), some still in their original leg
// shape. This did not exist before: keyframe_align/wheel_align_motion's
// Geometry assumes every leg is the same wheel shape, which is true for their
// phase (Phase 1, wheel-base alignment) but not for this one.
//
// The joint-chain constants (p1, p3, r1/r2/r3, from wheel_align_geometry_data.hpp)
// encode the physical abduction/hip/hub joints up to the shin's mount point --
// that kinematic chain doesn't change with the shin's material/shape. Only the
// distal endpoint model below differs: a wheel is modeled as a capsule
// (wheel_radius_m about a center offset along the mount's local Z, matching
// the existing trained geometry); a leg is modeled as a single point at
// leg_tip_hub_distance_m along that same local Z (a leg tip has no meaningful
// capsule radius the way a wheel tread does).
struct Geometry {
  static Mat mul(const Mat& a, const Mat& b) {
    Mat r{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) r[3 * i + j] += a[3 * i + k] * b[3 * k + j];
    return r;
  }
  static Vec mv(const Mat& a, const Vec& b) {
    Vec r{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r[i] += a[3 * i + j] * b[j];
    return r;
  }
  static Mat rz(double q) { return {std::cos(q), -std::sin(q), 0, std::sin(q), std::cos(q), 0, 0, 0, 1}; }
  static double norm(const Vec& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

  // Signed distance from `leg`'s shin-tip to the lowest other shin-tip
  // ("the support envelope"), along the measured gravity direction. Positive
  // means `leg` is that far below the current lowest support point -- i.e.
  // how much further it can descend before it becomes the new lowest point
  // (which is the wrong thing to be while it's still meant to be lifted, and
  // the right thing to be once it's meant to land).
  static double clearance_below_support(const std::array<double, 12>& q, const Vec& gravity,
                                         int leg, int converted) {
    using namespace neural_controller::align_geometry;
    double support = 1e3, active = 0;
    for (int k = 0; k < 4; ++k) {
      Mat a{}, b{}, c{};
      std::copy_n(r1.begin() + 9 * k, 9, a.begin());
      std::copy_n(r2.begin() + 9 * k, 9, b.begin());
      std::copy_n(r3.begin() + 9 * k, 9, c.begin());
      const auto r = mul(mul(mul(a, rz(q[3 * k])), b), rz(q[3 * k + 1]));
      const auto p = mv(r, {p3[3 * k], p3[3 * k + 1], p3[3 * k + 2]});
      const auto orientation = mul(r, c);
      const bool is_wheel = (converted >> k) & 1;
      const double extension = is_wheel ? wheel_center_z[k] : shin_reference::leg_tip_hub_distance_m;
      double z = 0, dot = 0;
      for (int i = 0; i < 3; ++i) {
        z -= (p[i] + p1[3 * k + i] + extension * orientation[3 * i + 2]) * gravity[i];
        dot -= orientation[3 * i + 2] * gravity[i];
      }
      const double radial = std::sqrt(std::max(1 - dot * dot, 0.));
      const double bottom = is_wheel
          ? z - shin_reference::wheel_radius_m * radial - shin_reference::wheel_half_width_m * std::abs(dot)
          : z - shin_reference::appendage_length_uncertainty_m;  // point tip: budget the uncertainty directly
      if (k == leg) active = bottom;
      else support = std::min(support, bottom);
    }
    return active - support;
  }
};
}  // namespace leg_to_wheel
