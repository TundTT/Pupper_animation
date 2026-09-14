#pragma once
#include "lift_request.hpp"
#include "geometry.hpp"
namespace notebook_alignment_v4 {
struct LiftSensors {
 std::array<double,12> q{},qd{};
 std::array<double,3> angular{},gravity{0,0,-1};
};
struct LiftCore {
 PositionTargets targets;
 ObservationHistory history;
 LiftRequest request;
 std::array<double,4> home{},hold{};
 std::array<double,12> command{},estimated_pd{};
 std::array<double,4> bottoms{};
 bool supported=false;
 double stand_settle=0;
 static bool finite(const LiftSensors& s){
  for(auto x:s.q)if(!std::isfinite(x))return false;
  for(auto x:s.qd)if(!std::isfinite(x))return false;
  for(auto x:s.angular)if(!std::isfinite(x))return false;
  for(auto x:s.gravity)if(!std::isfinite(x))return false;
  return true;
 }
 bool support(const LiftSensors& s) {
  bottoms=WheelGeometry::bottoms(s.q,s.gravity);
  auto mm=std::minmax_element(bottoms.begin(),bottoms.end());
  if(*mm.second-*mm.first>=.006 || -s.gravity[2]<=std::cos(.12))return false;
  double spin=0;for(auto v:s.angular)spin+=v*v;if(spin>=.3*.3)return false;
  for(auto v:s.qd)if(std::abs(v)>=.1)return false;
  for(int i=0;i<8;++i)if(std::abs(s.q[PositionTargets::proximal[i]]-targets.applied[i])>=.12)return false;
  for(int k=0;k<4;++k)if(std::abs(s.q[PositionTargets::hubs[k]]-hold[k])>=.035)return false;
  return true;
 }
 ObservationInput input(const LiftSensors& s)const{
  ObservationInput in;in.leg=request.leg;in.phase=request.actor_phase();in.desired_clearance=request.height;
  in.applied=targets.applied;in.target_velocity=targets.velocity;
  for(int i=0;i<12;++i){in.position[i]=s.q[i];in.velocity[i]=s.qd[i];}
  for(int i=0;i<3;++i){in.angular[i]=s.angular[i];in.gravity[i]=s.gravity[i];}
  for(int k=0;k<4;++k){in.home[k]=home[k];in.reference[k]=hold[k];}
  // Lift-only: upcoming hub speeds and rotation permission remain exactly zero.
  return in;
 }
 void reset(const LiftSensors& s,const std::array<double,4>& startup_home){
  if(!finite(s))throw std::invalid_argument("Invalid entry sensors");
  request.reset();stand_settle=0;home=startup_home;
  PositionTargets::V8 current{};
  for(int a=0;a<8;++a){current[a]=s.q[PositionTargets::proximal[a]];
   if(std::abs(current[a]-PositionTargets::nominal[a])>.25)throw std::invalid_argument("Lift entry requires near-nominal proximal stand");}
  for(int k=0;k<4;++k){if(!std::isfinite(home[k]))throw std::invalid_argument("Invalid startup hub home");hold[k]=s.q[PositionTargets::hubs[k]];}
  targets.reset(current);command=s.q;estimated_pd.fill(0);supported=support(s);
  if(!supported)throw std::invalid_argument("Lift entry requires stationary, level supported geometry");
  history.reset(input(s));
 }
 void prepare(double dt,int command_event,const LiftSensors& s){
  supported=support(s);
  stand_settle=supported?stand_settle+dt:0;
  request.advance(dt,command_event,supported && (request.phase!=0 || stand_settle>=.5));
  if(request.recovery_failed)throw std::runtime_error("Supported recovery did not settle");
  history.update(input(s));
 }
 void execute(double dt,const LiftSensors& s){
  std::array<float,4> wheel{};for(int k=0;k<4;++k)wheel[k]=hold[k];
  auto result=targets.step(dt,wheel);
  for(int i=0;i<12;++i)command[i]=(i%3==2)?hold[i/3]:result[i];
  for(int a=0;a<8;++a){int i=PositionTargets::proximal[a];if(s.q[i]<PositionTargets::lower[a]-.002||s.q[i]>PositionTargets::upper[a]+.002)throw std::runtime_error("Proximal measured limit violation");}
  for(int i=0;i<12;++i){
   estimated_pd[i]=PositionTargets::kp[i]*(command[i]-s.q[i])-PositionTargets::kd[i]*s.qd[i];
   // Motor firmware's total-torque clamp is unverified. Do not claim the
   // feedforward-effort clamp limits PD: stop on an over-limit sampled estimate.
   if(!std::isfinite(estimated_pd[i])||std::abs(estimated_pd[i])>3.)throw std::runtime_error("Estimated position-PD command exceeds3Nm");
  }
 }
};
}
