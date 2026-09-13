#pragma once
#include "neural_controller/inverted_triangle.hpp"

namespace joint_pose {
using inverted_triangle::Pose;
using inverted_triangle::low;
using inverted_triangle::high;
enum State { RAMP, UPPERS, HUBS, HOLD, FAULT };
enum Fault { NONE, INPUT, PERIOD, TRACKING, SPEED, RECOVERY, TIMEOUT, STOP };
struct Core {
  Pose start{},reference{},goal{},kp{},kd{},initial{};
  State state=FAULT;Fault fault=NONE;
  double elapsed=0,total=0,duration=0,gain=0,error=0,stable=0;
  // Software recovery allowance around policy command limits, not mechanical limits.
  static constexpr double recovery_margin=.35,torque_limit=.6;
  void stop(Fault f){if(state==FAULT&&fault!=NONE)return;fault=f;state=FAULT;kp.fill(0);kd.fill(0);gain=0;}
  static double seconds(const Pose&a,const Pose&b,bool hubs) {
    double t=hubs?12.:4.;
    for(int i=0;i<12;++i)if((i%3==2)==hubs) {
      const double d=std::abs(b[i]-a[i]),v=hubs?.5:.3,acc=hubs?1.2:.8;
      t=std::max({t,1.875*d/v,std::sqrt(5.773502691896258*d/acc)});
    }
    return t;
  }
  void reset(const Pose&q,const Pose&v,const Pose&target) {
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i])||!std::isfinite(v[i])||!std::isfinite(target[i])||std::abs(v[i])>.2 ||
         target[i]<low[i]||target[i]>high[i]||q[i]<low[i]-recovery_margin||q[i]>high[i]+recovery_margin)
        throw std::runtime_error("Invalid target or starting pose outside bounded recovery range");
      if(i%3==2 && std::abs(target[i]-q[i])>3.2)throw std::runtime_error("Hub move exceeds one half-turn; select explicit nearby winding");
      if(i%3!=2 && std::abs(target[i]-q[i])>1.)throw std::runtime_error("Upper move exceeds one radian");
      initial[i]=q[i];reference[i]=std::clamp(q[i],low[i],high[i]);
    }
    start=reference;goal=target;kp.fill(0);kd.fill(0);
    state=RAMP;fault=NONE;elapsed=total=stable=gain=error=0;duration=2.;
  }
  void step(double dt,const Pose&q,const Pose&v) {
    if(state==FAULT)return;
    if(!std::isfinite(dt)||dt<=0||dt>.02){stop(PERIOD);return;}
    elapsed+=dt;total+=dt;error=0;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i])||!std::isfinite(v[i])){stop(INPUT);return;}
      if(std::abs(v[i])>1.5){stop(SPEED);return;}
      const double lower=std::min(initial[i],low[i])-.015,upper=std::max(initial[i],high[i])+.015;
      if(q[i]<lower || q[i]>upper){stop(RECOVERY);return;}
    }
    if(state==RAMP) {
      gain=inverted_triangle::smooth(elapsed/2.);
      if(elapsed>=2){state=UPPERS;elapsed=0;start=reference;duration=seconds(start,goal,false);}
    } else if(state==UPPERS || state==HUBS) {
      const bool hubs=state==HUBS;const double u=inverted_triangle::smooth(elapsed/duration);
      for(int i=0;i<12;++i)if((i%3==2)==hubs)reference[i]=start[i]+u*(goal[i]-start[i]);
      if(elapsed>=duration) {
        bool settled=true;
        for(int i=0;i<12;++i)if((i%3==2)==hubs)
          settled=settled&&std::abs(q[i]-goal[i])<.06&&std::abs(v[i])<.15;
        stable=settled?stable+dt:0.;
        if(stable>=.4){state=hubs?HOLD:HUBS;elapsed=stable=0;start=reference;duration=seconds(start,goal,true);}
        else if(elapsed>duration+8){stop(TIMEOUT);return;}
      }
    }
    for(int i=0;i<12;++i) {
      const double e=reference[i]-q[i];error=std::max(error,std::abs(e));
      if(std::abs(e)>.45){stop(TRACKING);return;}
      const double base_kp=i%3==2?4.:5.,base_kd=i%3==2?.15:.25;
      // Conservative bound includes both PD terms: no software saturation of position error.
      const double demand=base_kp*std::abs(e)+base_kd*std::abs(v[i]);
      const double scale=std::min(gain,demand>0?torque_limit/demand:gain);
      kp[i]=base_kp*scale;kd[i]=base_kd*scale;
    }
  }
};
}
