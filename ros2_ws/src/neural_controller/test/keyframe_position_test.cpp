#include "neural_controller/keyframe_align/controller.hpp"
#include <iostream>

using namespace keyframe_align;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}

int main(){
  try {
    constexpr double dt=1./520;
    for(double turns:{0.,16*M_PI,-16*M_PI}){
      Controller c;c.config.rotation_floor_clearance_m=.005;
      V12 q{1,0,3.13+turns,-1,0,-3.13+turns,1,0,.4+turns,-1,0,-.4+turns},qd{};
      const std::array<double,4> home{q[2],q[5],q[8],q[11]};
      c.reset(q,home);
      auto advance=[&](int request,const V3& gravity=V3{0,0,-1}){
        const auto o=c.step(dt,request,q,qd,{0,0,0},gravity);
        require(o.authority,"Unexpected core fault");
        require(std::abs(o.integral)<=c.config.wheel_integral_limit_nm,"Integral torque remains bounded");
        for(int k=0;k<4;++k){require(o.wheel[k]==0,"No speed request in position mode");
          qd[3*k+2]=(o.wheel_position[k]-q[3*k+2])/dt;q[3*k+2]=o.wheel_position[k];}
        for(int a=0;a<8;++a){qd[rows[a]]=(o.position[a]-q[rows[a]])/dt;q[rows[a]]=o.position[a];}
        return o;
      };
      auto first=advance(0);
      for(int k=0;k<4;++k)require(first.wheel_position[k]==home[k],"No wrapped/zero angle jump at entry");
      for(int n=0;n<2300;++n)advance(0);
      for(int command=1;command<=4;++command){
        const int k=legs[command];bool disturbed=false,gate_checked=false;
        double previous=q[3*k+2],previous_rate=0;
        for(int n=0;n<30000;++n){
          const auto o=advance(command);
          const double rate=(o.wheel_position[k]-previous)/dt;
          if(o.phase==ROTATE && o.blocked==0){
            require(std::abs(rate)<=c.config.values[10]+1e-5,"Angle ramp exceeds speed limit");
            require(std::abs(rate-previous_rate)/dt<=c.config.values[11]+1e-3,"Angle ramp exceeds acceleration limit");
            if(!gate_checked && std::abs(o.error)<2.5){
              const auto before=q;
              auto blocked=advance(command,{std::sin(.2),0,-std::cos(.2)});
              require(blocked.blocked&TILT,"Tilt blocks rotation");
              require(blocked.wheel_position[k]==before[3*k+2],"Blocked gate discards pending angle demand");
              gate_checked=true;previous=q[3*k+2];previous_rate=0;continue;
            }
            if(!disturbed && std::abs(o.error)<.001 && std::abs(rate)<.005){
              // Reproduce the observed rear-wheel overshoot with zero speed.
              const double target=o.wheel_position[k];q[3*k+2]=target+.051;qd[3*k+2]=0;
              const auto correction=c.step(dt,command,q,qd,{0,0,0},{0,0,-1});
              require(correction.phase==ROTATE,"Must not lower while outside angle tolerance");
              require(c.config.wheel_position_kp*(correction.wheel_position[k]-q[3*k+2])<-.09,
                      "Overshoot must immediately produce reverse P correction");
              q[3*k+2]=correction.wheel_position[k];disturbed=true;previous=correction.wheel_position[k];previous_rate=(correction.wheel_position[k]-o.wheel_position[k])/dt;continue;
            }
          }
          previous=o.wheel_position[k];previous_rate=rate;
          if(c.completed&(1<<k))break;
        }
        require(disturbed&&gate_checked,"Must exercise overshoot and gate pause on every wheel");
        require(c.completed&(1<<k),"Each wheel must automatically lower and finish");
        require(std::abs(Controller::wrap(home[k]+M_PI-q[3*k+2]))<.025,"Calibrated final angle");
      }
      require(!c.step(dt,0,q,qd,{0,0,0},{0,0,-1},true).authority,"Stop removes authority");
    }
    // Frozen near-target encoder models a wheel that the P correction cannot
    // dislodge. The bias must build, remain capped, and clear on unsafe/stale use.
    Controller c;c.config.wheel_position_kp=4;
    V12 q{},qd{};for(int a=0;a<8;++a)q[rows[a]]=c.config.poses[2][a];
    q[8]=M_PI+.028;const std::array<double,4> home{};
    c.reset(q,home);c.phase=ROTATE;c.active=3;
    Output o;
    for(int n=0;n<6000;++n){
      o=c.step(dt,3,q,qd,{0,0,0},{0,0,-1});
      if(std::abs(o.wheel_position[2]-M_PI)>1e-9)
        require(o.wheel_effort[2]==0,"No integral accumulation during angle ramp");
      require(std::abs(o.wheel_effort[2])<=.10,"Integral torque cap");
      for(int k:{0,1,3})require(o.wheel_effort[k]==0,"Nudge only the requested wheel");
    }
    require(o.phase==ROTATE&&o.wheel_effort[2]<-.099,"Persistent small error builds reverse torque");
    auto crossing=c;auto opposite=q;opposite[8]=M_PI-.028;
    require(crossing.step(dt,3,opposite,qd,{0,0,0},{0,0,-1}).wheel_effort[2]>0,"Crossing discards opposing integral");
    auto blocked=c;
    require(blocked.step(dt,3,q,qd,{0,0,0},{std::sin(.2),0,-std::cos(.2)}).wheel_effort[2]==0,"Gate closure clears integral");
    auto distant=c;auto far=q;far[8]=M_PI+.2;
    require(distant.step(dt,3,far,qd,{0,0,0},{0,0,-1}).wheel_effort[2]==0,"Large error clears nudge");
    auto cancelled=c;
    require(cancelled.step(dt,0,q,qd,{0,0,0},{0,0,-1}).wheel_effort[2]==0,"Cancelled descent clears integral");
    auto stopped=c;
    require(stopped.step(dt,3,q,qd,{0,0,0},{0,0,-1},true).wheel_effort[2]==0,"Stop clears integral");
    auto accepted=c;auto near=q;near[8]=M_PI+.01;
    for(int n=0;n<280;++n)o=accepted.step(dt,3,near,qd,{0,0,0},{0,0,-1});
    require(o.phase==LOWER&&o.wheel_effort[2]<-.099,"Successful lowering retains bounded holding bias");
    accepted.reset(near,home);
    require(accepted.step(dt,0,near,qd,{0,0,0},{0,0,-1}).wheel_effort[2]==0,"Reactivation clears integral");
    std::cout<<"PASS: angle trajectories, all-wheel lowering, bounded near-target integral, crossing/gate/cancel/stop resets\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

