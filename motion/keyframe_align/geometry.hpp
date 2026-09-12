#pragma once
#include "neural_controller/wheel_align_motion.hpp"
#include "model_geometry.hpp"
namespace keyframe_align {
// Keyframe-only geometry: old policy estimates remain unchanged for historical replay.
struct Geometry : neural_controller::WheelAlignMotion {
 static std::array<double,3> margins(const std::array<double,12>& q,const Vec& gravity,int leg) {
  using namespace neural_controller::align_geometry;
  using namespace model_geometry;
  constexpr double radius=.0505,half=.01675;const double bound=std::hypot(radius,half);
  std::array<Vec,4> centers{};double support=1e3,active=0,wheel_gap=1e3,body_gap=1e3;
  for(int k=0;k<4;++k){
   Mat a{},b{},c{};std::copy_n(r1.begin()+9*k,9,a.begin());std::copy_n(r2.begin()+9*k,9,b.begin());std::copy_n(r3.begin()+9*k,9,c.begin());
   const auto r=mul(mul(mul(a,rz(q[3*k])),b),rz(q[3*k+1]));const auto p=mv(r,{p3[3*k],p3[3*k+1],p3[3*k+2]});const auto orientation=mul(r,c);
   double z=0,dot=0;
   for(int i=0;i<3;++i){centers[k][i]=p[i]+p1[3*k+i]+wheel_offset*orientation[3*i+2];z-=centers[k][i]*gravity[i];dot-=orientation[3*i+2]*gravity[i];}
   const double radial=std::sqrt(std::max(1-dot*dot,0.));
   if(k==leg)active=z-radius*radial-half*std::abs(dot);else support=std::min(support,z-.0455*radial-half*std::abs(dot));
   for(int b=0;b<box_count;++b){Vec local{};for(int i=0;i<3;++i){
    for(int j=0;j<3;++j)local[i]+=(centers[k][j]-box_positions[3*b+j])*box_rotations[9*b+3*j+i];
    local[i]=std::max(std::abs(local[i])-box_sizes[3*b+i],0.);
   }body_gap=std::min(body_gap,norm(local)-bound);}
  }
  for(int a=0;a<4;++a)for(int b=a+1;b<4;++b){Vec d{};for(int j=0;j<3;++j)d[j]=centers[a][j]-centers[b][j];wheel_gap=std::min(wheel_gap,norm(d)-2*bound);}
  return {active-support,wheel_gap,body_gap};
 }
};
}
