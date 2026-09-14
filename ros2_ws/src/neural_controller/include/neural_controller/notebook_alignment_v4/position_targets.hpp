#pragma once
// Versioned execution contract for the explicitly selected notebook lift trial.
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace notebook_alignment_v4 {
struct PositionTargets {
 using V8=std::array<float,8>;using V12=std::array<float,12>;
 static constexpr const char* contract="quadmorph-align-notebook-v4";
 static constexpr V8 nominal{1,0,-1,0,1,0,-1,0};
 static constexpr V8 lower{-1.12f,-.32f,-2.41f,-3.04f,-1.12f,-.32f,-2.41f,-3.04f};
 static constexpr V8 upper{2.41f,3.04f,1.12f,.32f,2.41f,3.04f,1.12f,.32f};
 static constexpr std::array<int,8> proximal{0,1,3,4,6,7,9,10};
 static constexpr std::array<int,4> hubs{2,5,8,11};
 static constexpr V12 kp{5,5,8,5,5,8,5,5,8,5,5,8};
 static constexpr V12 kd{.25f,.25f,1,.25f,.25f,1,.25f,.25f,1,.25f,.25f,1};
 V8 applied=nominal,velocity{},requested=nominal;
 void reset(const V8& current_command,const V8& current_target_velocity={}) {
  for(int i=0;i<8;++i) if(!std::isfinite(current_command[i])||current_command[i]<lower[i]||current_command[i]>upper[i]||!std::isfinite(current_target_velocity[i])||std::abs(current_target_velocity[i])>.100001f)throw std::invalid_argument("Invalid supported entry state");
  applied=requested=current_command;velocity=current_target_velocity;
 }
 void set_action(const V8& action) {
  for(int i=0;i<8;++i){if(!std::isfinite(action[i])||std::abs(action[i])>1.000001f)throw std::invalid_argument("Invalid actor output");requested[i]=std::clamp(nominal[i]+.75f*action[i],lower[i]+.06f,upper[i]-.06f);}
 }
 // Call each command update with measured dt; nominal simulation dt=1/520s.
 // Actor updates requests at52Hz. Phase changes never reset applied/velocity.
 V12 step(float dt,const std::array<float,4>& hub_reference) {
  if(!std::isfinite(dt)||dt<=0||dt>.01f)throw std::invalid_argument("Stale/invalid command interval");
  V12 command{};
  for(int i=0;i<8;++i){float e=requested[i]-applied[i];float sign=(e>0)-(e<0);float wanted=sign*std::min(.1f,std::sqrt(4.f*std::abs(e)));velocity[i]=std::clamp(wanted,velocity[i]-2.f*dt,velocity[i]+2.f*dt);applied[i]+=velocity[i]*dt;command[proximal[i]]=applied[i];}
  for(int k=0;k<4;++k){if(!std::isfinite(hub_reference[k]))throw std::invalid_argument("Invalid hub reference");command[hubs[k]]=hub_reference[k];}
  return command;
 }
};
}
