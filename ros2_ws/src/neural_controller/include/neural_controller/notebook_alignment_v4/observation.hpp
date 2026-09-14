#pragma once
#include "position_targets.hpp"
namespace notebook_alignment_v4 {
// Raw actor inputs for v4. Normalization is folded into the exported network.
// No contact/force/base-height/base-velocity truth is accepted here.
struct ObservationInput {
  std::array<float,3> angular{},gravity{0,0,-1};
  PositionTargets::V12 position{},velocity{};
  PositionTargets::V8 applied=PositionTargets::nominal,target_velocity{};
  std::array<float,4> home{},reference{},hub_velocity{};
  int leg=0,phase=0;
  float desired_clearance=0;
  bool rotation_enabled=false;
};
class ObservationHistory {
 public:
  std::array<float,288> values{};
  static std::array<float,72> frame(const ObservationInput& in) {
    if(in.leg<0||in.leg>3||in.phase<0||in.phase>3)throw std::invalid_argument("Invalid actor command/phase");
    std::array<float,72> x{};
    for(int i=0;i<3;++i){x[i]=in.angular[i];x[3+i]=in.gravity[i];}
    x[6+in.leg]=1;x[10+in.phase]=1;
    for(int i=0;i<8;++i){x[14+i]=in.position[PositionTargets::proximal[i]]-PositionTargets::nominal[i];x[42+i]=(in.applied[i]-PositionTargets::nominal[i])/.75f;}
    for(int k=0;k<4;++k){
      float q=in.position[PositionTargets::hubs[k]],relative=q-in.home[k],error=in.reference[k]-q;
      x[22+k]=std::sin(relative);x[26+k]=std::cos(relative);
      x[52+k]=in.hub_velocity[k];x[56+k]=std::sin(error);x[60+k]=std::cos(error);
    }
    for(int i=0;i<12;++i)x[30+i]=in.velocity[i]*.05f;
    for(int i=0;i<8;++i)x[64+i]=in.target_velocity[i];
    x[50]=in.desired_clearance;x[51]=in.rotation_enabled?1.f:0.f;
    for(float value:x)if(!std::isfinite(value))throw std::invalid_argument("Non-finite actor sensor/state input");
    return x;
  }
  void reset(const ObservationInput& in){auto x=frame(in);for(int h=0;h<4;++h)std::copy(x.begin(),x.end(),values.begin()+72*h);}
  void update(const ObservationInput& in){auto x=frame(in);std::copy_backward(values.begin(),values.begin()+216,values.end());std::copy(x.begin(),x.end(),values.begin());}
  static int leg_for_external_command(int command){
    constexpr std::array<int,5> map{-1,1,0,2,3};
    if(command<0||command>4)throw std::invalid_argument("Invalid external command");
    return map[command]; // -1 means stand: retain the active leg and use phase0.
  }
};
}
