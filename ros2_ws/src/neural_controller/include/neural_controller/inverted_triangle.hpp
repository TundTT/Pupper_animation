#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace inverted_triangle {
using Pose = std::array<double, 12>;
inline constexpr double hz = 520.;
inline constexpr double tilt_limit = 8. * 3.14159265358979323846 / 180.;
inline constexpr Pose low{-1.12,-.32,-1000,-2.41,-3.04,-1000,-1.12,-.32,-1000,-2.41,-3.04,-1000};
inline constexpr Pose high{2.41,3.04,1000,1.12,.32,1000,2.41,3.04,1000,1.12,.32,1000};
inline double smooth(double u) { u=std::clamp(u,0.,1.); return u*u*u*(10+u*(-15+6*u)); }
struct Segment { int stage=0; std::string phase; int steps=0; Pose target{}; };
enum State { RAMP=0, READY=1, RUNNING=2, PAUSED=3, DONE=4, FAULT=5 };
enum Fault { NONE=0, TIMING=1, SENSORS=2, TILT=3, TRACKING=4, SPEED=5, TORQUE=6,
             OPERATOR=7, SETTLE_TIMEOUT=8, ACTIVATION=9, COMMAND=10 };

// No model inference, geometry estimate, contact-force input or file I/O in step().
// Commands 1..4 authorize one complete leg each. Early commands are discarded.
struct Controller {
  Pose initial{}, kp{}, kd{}, offset{}, command{}, entry{}, previous{};
  std::vector<Segment> segments;
  State state=FAULT;
  Fault fault=ACTIVATION;
  int segment=0, completed=0;
  double elapsed=0, ramp=0, gate_wait=0, gain=0, max_error=0, max_torque=0;

  void validate() const {
    if(segments.size()!=38) throw std::runtime_error("Expected reviewed 38-segment plan");
    Pose prior=initial; int total=0, stage=0;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(initial[i]) || initial[i]<low[i] || initial[i]>high[i] ||
         kp[i]!=(i%3==2 ? 4.:5.) || kd[i]!=(i%3==2 ? .15:.25))
        throw std::runtime_error("Initial pose/gains differ from motion contract");
    }
    for(const auto& s:segments) {
      if(s.stage<stage || s.stage>stage+1 || s.stage>3 || s.steps<=0 || s.steps>20000)
        throw std::runtime_error("Invalid phase order/duration");
      stage=s.stage; total+=s.steps;
      for(int i=0;i<12;++i) {
        const double delta=std::abs(s.target[i]-prior[i]), seconds=s.steps/hz;
        const double speed=i%3==2 ? .5 : (i%3==0 ? .45:.65);
        const double accel=i%3==2 ? 1.2:2.;
        if(!std::isfinite(s.target[i]) || s.target[i]<low[i] || s.target[i]>high[i] ||
           1.875*delta/seconds>speed+1e-9 || (10/std::sqrt(3))*delta/(seconds*seconds)>accel+1e-9)
          throw std::runtime_error("Target exceeds position/speed/acceleration envelope");
      }
      prior=s.target;
    }
    if(total!=67278 || stage!=3) throw std::runtime_error("Unexpected plan length");
  }
  void stop(Fault reason) { if(state!=FAULT) fault=reason; state=FAULT; gain=0; }
  void reset(const Pose& q,const Pose& mapping) {
    validate(); offset=mapping;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i]) || !std::isfinite(offset[i]) ||
         (i%3!=2 && offset[i]!=0.) || std::abs(q[i]-offset[i]-initial[i])>.03)
        throw std::runtime_error("Place robot at confirmed inverted start; no automatic approach");
      for(const auto& s:segments) if(s.target[i]+offset[i]<low[i] || s.target[i]+offset[i]>high[i])
        throw std::runtime_error("Mapped trajectory exceeds hardware limits");
      entry[i]=q[i]-offset[i];
    }
    command=entry; previous=initial; segment=0; completed=0; elapsed=0; ramp=0;
    gate_wait=0; gain=0; max_error=0; max_torque=0; state=RAMP; fault=NONE;
  }
  bool settled(const Pose& q,const Pose& velocity,int tight_hub=-1) const {
    for(int i=0;i<12;++i)
      if(std::abs(command[i]+offset[i]-q[i])>(i==tight_hub ? .035:(i%3==2 ? .15:.22)) || std::abs(velocity[i])>.1) return false;
    return true;
  }
  void step(double dt,const Pose& q,const Pose& velocity,double tilt,int request=0,bool fresh=false) {
    if(state==FAULT) return;
    if(!std::isfinite(dt) || dt<=0 || dt>.01) { stop(TIMING); return; }
    if(!std::isfinite(tilt)) { stop(SENSORS); return; }
    if(tilt>=tilt_limit) { stop(TILT); return; }
    if(fresh && (request<0 || request>4)) { stop(request==-1 ? OPERATOR:COMMAND); return; }
    max_error=0; max_torque=0;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i]) || !std::isfinite(velocity[i])) { stop(SENSORS); return; }
      const double error=std::abs(command[i]+offset[i]-q[i]);
      max_error=std::max(max_error,error);
      if(error>(i%3==2 ? .15:.25)) { stop(TRACKING); return; }
      if(std::abs(velocity[i])>2.) { stop(SPEED); return; }
    }
    if(state==RAMP) {
      ramp=std::min(2.,ramp+dt); gain=smooth(ramp/2.);
      for(int i=0;i<12;++i) command[i]=entry[i]+gain*(initial[i]-entry[i]);
      if(ramp>=2.) state=READY;
    } else if(state==READY || state==PAUSED) {
      if(fresh && request==completed+1 && settled(q,velocity)) {
        state=RUNNING; elapsed=0; gate_wait=0;
      }
    }
    // Pauses occur only on four-support planted endpoints, never at a lifted pose.
    if(state==RUNNING) {
      const auto& s=segments.at(segment);
      const double seconds=s.steps/hz;
      elapsed=std::min(seconds,elapsed+dt);
      const double blend=smooth(elapsed/seconds);
      for(int i=0;i<12;++i) command[i]=previous[i]+blend*(s.target[i]-previous[i]);
      if(elapsed>=seconds-1e-10) {
        const bool gate=s.phase=="clearance_hold" || s.phase=="angle_hold" || s.phase=="planted_hold";
        // Validated loaded support hubs deflect up to 0.103 rad nominally.
        // The original 0.035-rad final-angle gate is for the active hub only.
        constexpr std::array<int,4> active_hubs{8,11,2,5};
        if(gate && !settled(q,velocity,active_hubs[s.stage])) {
          gate_wait+=dt; if(gate_wait>3.) stop(SETTLE_TIMEOUT);
        } else {
          previous=s.target; elapsed=0; gate_wait=0; ++segment;
          if(s.phase=="planted_hold") {
            ++completed; state=completed==4 ? DONE:PAUSED;
          }
        }
      }
    }
    // Hardware effort_max clips feed-forward only. Reject excessive predicted PD
    // demand separately; this estimate is not a measured motor torque limit.
    for(int i=0;i<12 && state!=FAULT;++i) {
      const double tau=gain*(kp[i]*(command[i]+offset[i]-q[i])-kd[i]*velocity[i]);
      max_torque=std::max(max_torque,std::abs(tau));
      if(!std::isfinite(tau) || std::abs(tau)>1.5) stop(TORQUE);
    }
  }
};
}  // namespace inverted_triangle
