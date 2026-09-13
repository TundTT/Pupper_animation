#pragma once
#include "neural_controller/inverted_triangle.hpp"
#include <Eigen/Core>

namespace hub_roll {
using inverted_triangle::Pose;
using V3=Eigen::Vector3d;
inline constexpr Pose base_kp{5,5,4,5,5,4,5,5,4,5,5,4};
inline constexpr Pose base_kd{.25,.25,.15,.25,.25,.15,.25,.25,.15,.25,.25,.15};
// Suspended diagnostic in the live encoder frame. No geometric mapping,
// ground support estimate, proximal repositioning, or walking handoff.
struct Controller {
  Pose initial{},kp=base_kp,kd=base_kd,offset{},command{},entry{},goal{};
  struct {std::array<double,4> loads{};double elapsed=0;} support;
  inverted_triangle::State state=inverted_triangle::FAULT;
  inverted_triangle::Fault fault=inverted_triangle::ACTIVATION;
  int segment=0,completed=0,run_steps=0;
  double elapsed=0,ramp=0,gain=0,max_error=0,max_torque=0;
  void validate() const {if(kp!=base_kp || kd!=base_kd)throw std::runtime_error("Unexpected hub diagnostic gains");}
  void stop(inverted_triangle::Fault why){if(state!=inverted_triangle::FAULT)fault=why;state=inverted_triangle::FAULT;gain=0;}
  void reset(const Pose& q,const Pose& mapping) {
    using namespace inverted_triangle;
    kp=base_kp;kd=base_kd;validate();
    if(mapping!=Pose{})throw std::runtime_error("Hub diagnostic uses unchanged encoder coordinates");
    initial=entry=command=goal=q;offset.fill(0);
    for(int leg=0;leg<4;++leg)goal[3*leg+2]+=(leg%2==0?-M_PI:M_PI);
    for(int i=0;i<12;++i)if(!std::isfinite(q[i]) || std::min(q[i],goal[i])<low[i] || std::max(q[i],goal[i])>high[i])
      throw std::runtime_error("Hub diagnostic start/end exceeds joint limits");
    support={};run_steps=completed=segment=0;ramp=elapsed=gain=max_error=max_torque=0;
    state=RAMP;fault=NONE;
  }
  bool settled(const Pose& q,const Pose& qd) const {
    for(int i=0;i<12;++i)if(std::abs(qd[i])>.1 || std::abs(q[i]-command[i])>.1)return false;
    return true;
  }
  void step(double dt,const Pose& q,const Pose& qd,const V3& gravity,int request=0,bool fresh=false) {
    using namespace inverted_triangle;
    if(state==FAULT)return;
    if(!std::isfinite(dt)||dt<=0||dt>.01){stop(TIMING);return;}
    if(!gravity.allFinite()||std::abs(gravity.norm()-1)>.02){stop(SENSORS);return;}
    if(-gravity[2]<=std::cos(tilt_limit)){stop(TILT);return;}
    if(fresh&&(request<0||request>1)){stop(request==-1?OPERATOR:COMMAND);return;}
    max_error=max_torque=0;
    for(int i=0;i<12;++i){
      if(!std::isfinite(q[i])||!std::isfinite(qd[i])){stop(SENSORS);return;}
      const double error=std::abs(command[i]-q[i]);max_error=std::max(max_error,error);
      if(error>(i%3==2?.15:.25)||q[i]<low[i]||q[i]>high[i]){stop(TRACKING);return;}
      if(std::abs(qd[i])>2.){stop(SPEED);return;}
    }
    if(state==RAMP){ramp=std::min(2.,ramp+dt);gain=smooth(ramp/2.);if(ramp>=2.)state=READY;}
    else if(state==READY&&fresh&&request==1&&settled(q,qd)){state=RUNNING;run_steps=0;}
    if(state==RUNNING){
      const int k=run_steps++;elapsed=run_steps/hz;segment=k<1040?0:1;
      for(int i=0;i<12;++i)kd[i]=2*base_kd[i];
      if(k>=1040)for(int leg=0;leg<4;++leg){const int i=3*leg+2;command[i]=initial[i]+smooth((k-1039.)/6240.)*(goal[i]-initial[i]);}
      if(run_steps>=7280){command=goal;kd=base_kd;state=DONE;completed=1;segment=3;}
    }
    for(int i=0;i<12;++i){
      const double torque=gain*(kp[i]*(command[i]-q[i])-kd[i]*qd[i]);max_torque=std::max(max_torque,std::abs(torque));
      if(!std::isfinite(torque)||std::abs(torque)>1.5){stop(TORQUE);return;}
    }
  }
};
}
