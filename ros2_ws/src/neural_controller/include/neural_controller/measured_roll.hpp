#pragma once
#include "neural_controller/triangle_roll.hpp"

// Port of handoff_entry.py a2dd1633 and balanced_support.py a1f166f2 from
// W&B 360ffcd09344487d. Physical encoder offsets are fixed at activation;
// they are not estimated from an arbitrary robot pose.
namespace measured_roll {
using namespace inverted_triangle;
using triangle_roll::V3;
inline constexpr double pi=3.14159265358979323846;
inline constexpr Pose neutral=triangle_roll::goal;
inline constexpr Pose point_up=neural_controller::policy_home::with_hubs({pi-1,1-pi,pi-1,1-pi});
inline constexpr Pose speed{.45,.65,.5,.45,.65,.5,.45,.65,.5,.45,.65,.5};
inline constexpr Pose accel{2,2,1.2,2,2,1.2,2,2,1.2,2,2,1.2};
// Include the documented ~0.18 rad hanging hips without inferring new offsets.
inline double entry_radius(int joint) {return joint%3==1?.20:.15;}
inline double duration(const Pose& a,const Pose& b,double minimum) {
  for(int i=0;i<12;++i) {
    double d=std::abs(a[i]-b[i]);
    minimum=std::max({minimum,1.875*d/speed[i],std::sqrt((10/std::sqrt(3.))*d/accel[i])});
  }
  return minimum;
}
enum Phase { ENTRY,ROLL,SETTLE,FINISHED,FAILED };
struct Motion {
  Pose command{},velocity{},start{},entry{},goal{},integral{};
  Phase phase=ENTRY;
  double elapsed=0,stable=0,entry_seconds=2,roll_seconds=12;
  void reset(const Pose& q,const Pose& previous) {
    command=previous;start=previous;entry=neutral;goal=neutral;
    velocity.fill(0);integral.fill(0);elapsed=stable=0;phase=ENTRY;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i]) || !std::isfinite(previous[i]) || q[i]<low[i] || q[i]>high[i] ||
         previous[i]<low[i] || previous[i]>high[i] || std::abs(previous[i]-q[i])>.15)
        throw std::runtime_error("Invalid measured handoff state");
    }
    for(int leg=0;leg<4;++leg) {
      const int h=3*leg+2;const double sign=leg%2==0?-1.:1.;
      entry[h-1]=.1*sign;entry[h]=q[h];
      goal[h]+=2*pi*std::nearbyint((q[h]+sign*pi-goal[h])/(2*pi));
      if(std::abs(goal[h]-q[h]-sign*pi)>.35)
        throw std::runtime_error("Point-up hub reference required");
    }
    entry_seconds=duration(start,entry,2.);roll_seconds=duration(entry,goal,12.);
  }
  void step(const Pose& q,const Pose& qd,double dt,const Pose& correction) {
    if(phase==FINISHED || phase==FAILED)return;
    if(!std::isfinite(dt)||dt<=0||dt>.01)throw std::runtime_error("Invalid motion period");
    for(int i=0;i<12;++i)if(!std::isfinite(q[i])||!std::isfinite(qd[i])||
      !std::isfinite(correction[i])||std::abs(correction[i])>.200001)
        throw std::runtime_error("Invalid motion feedback");
    elapsed+=dt;Pose target{};
    if(phase==ENTRY) {
      bool good=true;
      for(int i=0;i<12;++i) {
        double ref=start[i]+smooth(elapsed/entry_seconds)*(entry[i]-start[i]);
        if(i%3!=2)integral[i]=std::clamp(integral[i]+.25*(ref-q[i])*dt,-.15,.15);
        target[i]=ref+integral[i];
        good=good && std::abs(q[i]-entry[i])<=.10 && std::abs(qd[i])<=.12;
      }
      stable=good && elapsed>=entry_seconds?stable+dt:0.;
      if(stable>=.25){phase=ROLL;elapsed=0;start=command;}
      else if(elapsed>entry_seconds+8)phase=FAILED;
    } else if(phase==ROLL) {
      for(int i=0;i<12;++i)target[i]=start[i]+smooth(elapsed/roll_seconds)*(goal[i]-start[i]);
      if(elapsed>=roll_seconds){phase=SETTLE;elapsed=0;}
    } else {
      for(int i=0;i<12;++i)target[i]=goal[i]+correction[i];
      if(elapsed>=32)phase=FINISHED;
    }
    Pose next{},next_velocity{};
    for(int i=0;i<12;++i) {
      target[i]=std::clamp(target[i],low[i],high[i]);
      const double wanted=std::clamp(5*(target[i]-command[i]),-speed[i],speed[i]);
      next_velocity[i]=velocity[i]+std::clamp(wanted-velocity[i],-accel[i]*dt,accel[i]*dt);
      next[i]=command[i]+dt*next_velocity[i];
      if(next[i]<low[i] || next[i]>high[i]){phase=FAILED;return;}
    }
    command=next;velocity=next_velocity;
  }
};

// Lifecycle envelope: activation holds the measured pose, START performs entry
// then roll. Suspended diagnostic omits the ground-load estimator and settles
// with the same bounded filter. DONE holds and cannot activate walking.
struct Controller {
  Pose initial=point_up,kp=triangle_roll::base_kp,kd=triangle_roll::base_kd;
  Pose offset{},command{},entry{};
  Motion motion;
  triangle_roll::Support support;
  bool stand_only=false;
  State state=FAULT;Fault fault=ACTIVATION;
  int segment=0,completed=0,run_steps=0;
  double elapsed=0,ramp=0,gain=0,max_error=0,max_torque=0;
  void validate() const {
    if(initial!=point_up || kp!=triangle_roll::base_kp || kd!=triangle_roll::base_kd)
      throw std::runtime_error("Unexpected measured-roll pose/gains");
  }
  void stop(Fault why){if(state!=FAULT)fault=why;state=FAULT;gain=0;}
  void reset(const Pose& q,const Pose& mapping) {
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i]) || !std::isfinite(mapping[i]) || q[i]<low[i] || q[i]>high[i] ||
         (i%3!=2 && (mapping[i]!=0. || std::abs(q[i]-neutral[i])>entry_radius(i))))
        throw std::runtime_error("Calibrated near-alignment proximal pose required");
      entry[i]=q[i]-mapping[i];
    }
    motion.reset(entry,entry);
    for(int i=0;i<12;++i) {
      double a=std::min({entry[i],motion.entry[i]-(i%3==2?0.:.15),motion.goal[i]-.2});
      double b=std::max({entry[i],motion.entry[i]+(i%3==2?0.:.15),motion.goal[i]+.2});
      if(a+mapping[i]<low[i] || b+mapping[i]>high[i])
        throw std::runtime_error("Mapped entry/roll/support exceeds hardware limits");
    }
    offset=mapping;command=entry;support.set_goal(motion.goal);
    kd=triangle_roll::base_kd;run_steps=completed=segment=0;
    elapsed=ramp=gain=max_error=max_torque=0;state=RAMP;fault=NONE;
  }
  bool settled(const Pose& q,const Pose& qd) const {
    for(int i=0;i<12;++i)if(std::abs(qd[i])>.1 || std::abs(q[i]-offset[i]-command[i])>.1)return false;
    return true;
  }
  void step(double dt,const Pose& encoder,const Pose& qd,const V3& gravity,int request=0,bool fresh=false) {
    if(state==FAULT)return;
    if(!std::isfinite(dt)||dt<=0||dt>.01){stop(TIMING);return;}
    if(!gravity.allFinite()||std::abs(gravity.norm()-1)>.02){stop(SENSORS);return;}
    if(-gravity[2]<=std::cos(tilt_limit)){stop(TILT);return;}
    if(fresh&&(request<0||request>1)){stop(request==-1?OPERATOR:COMMAND);return;}
    Pose q{};max_error=max_torque=0;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(encoder[i])||!std::isfinite(qd[i])){stop(SENSORS);return;}
      q[i]=encoder[i]-offset[i];double error=std::abs(command[i]-q[i]);max_error=std::max(max_error,error);
      if(error>(i%3==2?.15:.25)||encoder[i]<low[i]||encoder[i]>high[i]){stop(TRACKING);return;}
      if(std::abs(qd[i])>2.){stop(SPEED);return;}
    }
    if(state==RAMP) {
      ramp=std::min(2.,ramp+dt);gain=smooth(ramp/2.);
      if(ramp>=2.)state=READY;
    } else if(state==READY && fresh && request==1 && settled(encoder,qd)) {
      // Re-enter from the live measured pose while preserving the held command.
      try { motion.reset(q,command);support.set_goal(motion.goal); }
      catch(const std::exception&) {stop(ACTIVATION);return;}
      state=RUNNING;run_steps=0;
    }
    if(state==RUNNING) {
      if(motion.phase==SETTLE && run_steps%10==0 && !stand_only)
        support.update(q,qd,command,gravity,10./hz);
      try { motion.step(q,qd,1./hz,stand_only?Pose{}:support.offset); }
      catch(const std::exception&) {stop(SENSORS);return;}
      ++run_steps;elapsed=run_steps/hz;segment=static_cast<int>(motion.phase);
      if(motion.phase==FAILED){stop(SETTLE_TIMEOUT);return;}
      command=motion.command;
      for(int i=0;i<12;++i)kd[i]=triangle_roll::base_kd[i]*(motion.phase==ENTRY || motion.phase==ROLL?2.:1.);
      if(motion.phase==FINISHED){state=DONE;completed=1;}
    }
    for(int i=0;i<12;++i) {
      const double torque=gain*(kp[i]*(command[i]-q[i])-kd[i]*qd[i]);max_torque=std::max(max_torque,std::abs(torque));
      if(!std::isfinite(torque)||std::abs(torque)>1.5){stop(TORQUE);return;}
    }
  }
};
}
