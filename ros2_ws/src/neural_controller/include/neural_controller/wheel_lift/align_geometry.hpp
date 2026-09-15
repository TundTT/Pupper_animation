#pragma once
#include "geometry.hpp"
#include <stdexcept>
namespace wheel_lift {
// Same model as WheelGeometry; conservative cylinder spheres and base collision
// box. Heating-backpack exclusion is explicit; it does not exclude the body.
struct AlignGeometry : WheelGeometry {
 static constexpr V3 box_pos={0.02146,0,0.033450000000000001};
 static constexpr V3 box_size={0.045069999999999999,0.063789999999999999,0.129715};
 static constexpr M3 box_rotation={-3.9999999999484892e-06,3.9999999999484892e-06,-0.99999999998400013,0.99999999999200018,7.9999340485414905e-12,-3.9999999999484892e-06,-8.000045070843953e-12,-0.99999999999200018,-3.9999999998929781e-06};
 static std::array<double,3> margins(const std::array<double,12>& q,const V3& gravity,int leg){
  if(leg<0||leg>3)throw std::invalid_argument("Invalid alignment leg");
  std::array<V3,bodies.size()> p{};std::array<M3,bodies.size()> R{};
  R[0]=R[1]=M3{1,0,0,0,1,0,0,0,1};
  for(unsigned b=2;b<bodies.size();++b){auto t=bodies[b];M3 local=t.rotation;V3 position=t.pos;
   if(t.joint>=0){double co=std::cos(q[t.joint]),si=std::sin(q[t.joint]);M3 spin{co,-si,0,si,co,0,0,0,1};auto offset=mv(spin,t.pivot);for(int i=0;i<3;++i)offset[i]=t.pivot[i]-offset[i];position=add(position,mv(local,offset));local=mul(local,spin);}
   p[b]=add(p[t.parent],mv(R[t.parent],position));R[b]=mul(R[t.parent],local);
  }
  std::array<V3,4> centers{};double active=0,support=1e3,wheel_gap=1e3,body_gap=1e3;
  constexpr double radius=.0505,half=.01675;const double sphere=std::hypot(radius,half);
  for(int k=0;k<4;++k){int b=wheel_body[k];centers[k]=add(p[b],mv(R[b],wheel_pos[k]));auto orient=mul(R[b],wheel_rotation[k]);double z=0,az=0;
   for(int i=0;i<3;++i){z-=centers[k][i]*gravity[i];az-=orient[3*i+2]*gravity[i];}
   double radial=std::sqrt(std::max(0.,1-az*az));
   if(k==leg)active=z-radius*radial-half*std::abs(az);
   else support=std::min(support,z-.0455*radial-half*std::abs(az));
   double squared=0;for(int i=0;i<3;++i){double local=0;for(int j=0;j<3;++j)local+=(centers[k][j]-box_pos[j])*box_rotation[3*j+i];local=std::max(std::abs(local)-box_size[i],0.);squared+=local*local;}
   body_gap=std::min(body_gap,std::sqrt(squared)-sphere);
  }
  for(int k=0;k<4;++k)for(int j=k+1;j<4;++j){double squared=0;for(int i=0;i<3;++i){double d=centers[k][i]-centers[j][i];squared+=d*d;}wheel_gap=std::min(wheel_gap,std::sqrt(squared)-2*sphere);}
  // Corrected floor bound: LOWEST support upper bound, never highest support.
  return {active-support,wheel_gap,body_gap};
 }
 static bool gate(const std::array<double,3>& m,const V3& gravity,const V3& angular){
  double spin=0;for(double w:angular)spin+=w*w;
  return m[0]>.010&&m[1]>.010&&m[2]>.005&&-gravity[2]>std::cos(.12)&&spin<.3*.3;
 }
};
}
