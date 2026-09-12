#pragma once
#include "neural_controller/wheel_align_motion.hpp"

namespace keyframe_align {
// Flat, rigid ground and the wheel model's radius interval are assumptions.
// Ground cannot lie ABOVE the lowest support-wheel bottom without penetration.
// The highest support bottom (legacy v5) instead follows an unloaded wheel and
// unnecessarily underestimates clearance. Use the lowest support upper bound
// and the active-wheel lower bound; keep the 10 mm rotation threshold unchanged.
struct Geometry : neural_controller::WheelAlignMotion {
  static std::array<double,3> margins(const std::array<double,12>& q,const Vec& gravity,int leg) {
    auto result=neural_controller::WheelAlignMotion::margins(q,gravity,leg);
    using namespace neural_controller::align_geometry;
    double support=1e3,active=0;
    for(int k=0;k<4;++k) {
      Mat a{},b{},c{};
      std::copy_n(r1.begin()+9*k,9,a.begin());std::copy_n(r2.begin()+9*k,9,b.begin());std::copy_n(r3.begin()+9*k,9,c.begin());
      auto r=mul(mul(mul(a,rz(q[3*k])),b),rz(q[3*k+1]));
      auto p=mv(r,{p3[3*k],p3[3*k+1],p3[3*k+2]});auto orientation=mul(r,c);
      double z=0,dot=0;
      for(int i=0;i<3;++i){z-=(p[i]+p1[3*k+i]+wheel_center_z[k]*orientation[3*i+2])*gravity[i];dot-=orientation[3*i+2]*gravity[i];}
      const double radial=std::sqrt(std::max(1-dot*dot,0.));
      // 48 mm nominal +/- 2.5 mm, matching modeled manufacturing variation.
      if(k==leg)active=z-.0505*radial-.01675*std::abs(dot);
      else support=std::min(support,z-.0455*radial-.01675*std::abs(dot));
    }
    result[0]=active-support;
    return result;
  }
};
}
