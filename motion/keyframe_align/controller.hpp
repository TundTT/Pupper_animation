#pragma once
#include "neural_controller/wheel_align_motion.hpp"
#include <stdexcept>

namespace keyframe_align {
using Geometry = neural_controller::WheelAlignMotion;
using V8 = std::array<double,8>;
using V12 = std::array<double,12>;
using V3 = std::array<double,3>;
inline constexpr std::array<int,8> rows{0,1,3,4,6,7,9,10};
inline constexpr std::array<int,5> legs{-1,1,0,2,3};
enum Phase { ENTRY, SHIFT, LIFT, ROTATE, LOWER, RECENTER, HOLD, STOPPED };
enum Block { TRAJECTORY=1, FLOOR=2, WHEEL=4, BODY=8, TILT=16, ANGULAR=32 };
struct Config {
  // Public ordering is also used by the small simulation C ABI.
  std::array<double,15> values{2.,2.,2.,3.,2.,48.,2.,.35,.8,.4,.5,1.2,.45,.65,2.};
  std::array<V8,4> poses{{
    {{1.16890458,1.0021502200000001,-1.2625986899999999,0.28754656000000001,1.0167859299999999,-0.16507621,-0.60319067999999998,-0.19620607000000001}},
    {{1.2625986899999999,-0.28754656000000001,-1.16890458,-1.0021502200000001,0.60319067999999998,0.19620607000000001,-1.0167859299999999,0.16507621}},
    {{1,0,-1,0,1,1.05,-1,0}},
    {{1,0,-1,0,1,0,-1,-1.05}},
  }};
  void validate() const {
    for(int i=0;i<15;++i) if(!std::isfinite(values[i])||values[i]<0 || (values[i]==0 && (i<6||i>9)))
      throw std::invalid_argument("Finite nonnegative gains and positive timing/limits required");
    if(values[5]<values[1]+values[2]+1 || values[9]>1 || values[10]>2 || values[11]>4 ||
       values[12]>1 || values[13]>1 || values[14]>4) throw std::invalid_argument("Unsafe timing or speed configuration");
    for(const auto &p:poses) for(int a=0;a<8;++a)
      if(!std::isfinite(p[a])||p[a]<Geometry::low[a]||p[a]>Geometry::high[a]) throw std::invalid_argument("Keyframe outside proximal limits");
  }
};
struct Output {
  V8 position{};
  std::array<double,4> wheel{};
  V3 margins{};
  int phase=ENTRY,active=0,completed=0,blocked=TRAJECTORY,timeout=-1;
  double error=0,up_seconds=0,integral=0;
  bool authority=false;
};
class Controller {
 public:
  Config config;
  Phase phase=STOPPED;
  int active=0,completed=0,timeout=-1;
  Output output;
  static double wrap(double x){return std::atan2(std::sin(x),std::cos(x));}
  static double norm(const V3 &v){return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
  static double smooth(double u){return Geometry::smooth(u);}
  void reset(const V12 &q,const std::array<double,4>& calibrated_home) {
    config.validate();
    for(double x:q) if(!std::isfinite(x)) throw std::invalid_argument("Invalid encoders");
    for(double x:calibrated_home) if(!std::isfinite(x)) throw std::invalid_argument("Invalid calibration");
    for(int a=0;a<8;++a) if(q[rows[a]]<Geometry::low[a]||q[rows[a]]>Geometry::high[a])
      throw std::invalid_argument("Initial proximal encoder outside limits");
    phase=ENTRY;active=completed=0;timeout=-1;elapsed=up_time=gate_time=settled=integral=wheel_speed=0;
    verified=false;velocity.fill(0);
    for(int a=0;a<8;++a) applied[a]=q[rows[a]];
    for(int k=0;k<4;++k){hold[k]=wrap(q[3*k+2]);target[k]=wrap(calibrated_home[k]+std::acos(-1.));}
    start=applied;endpoint=Geometry::neutral;output={};
  }
  Output step(double dt,int requested,const V12 &q,const V12 &qd,const V3 &angular,const V3 &gravity,bool stop=false) {
    bool finite=std::isfinite(dt)&&dt>0&&dt<=.04&&requested>=0&&requested<=4;
    for(double x:q)finite=finite&&std::isfinite(x);
    for(double x:qd)finite=finite&&std::isfinite(x);
    for(double x:angular)finite=finite&&std::isfinite(x);
    for(double x:gravity)finite=finite&&std::isfinite(x);
    finite=finite&&std::abs(norm(gravity)-1)<.02;
    if(stop||!finite||-gravity[2]<std::cos(.6))phase=STOPPED;
    if(phase==STOPPED){output.authority=false;output.phase=STOPPED;output.wheel.fill(0);return output;}
    // Timeout commands cannot silently retry. A stand/different request clears the latch.
    if(timeout>=0 && requested!=timeout) timeout=-1;
    auto is_up=[&](){return phase==SHIFT||phase==LIFT||phase==ROTATE;};
    if(is_up()){
      up_time+=dt;
      if(requested!=active || up_time>=config.values[5]){
        if(up_time>=config.values[5])timeout=active;
        verified=false;begin_lower(q);
      }
    }
    if(phase==HOLD && requested!=0 && requested!=timeout && !(completed&(1<<legs[requested]))) {
      active=requested;up_time=0;verified=false;integral=0;gate_time=settled=0;wheel_speed=0;
      wheel_reference=wrap(q[3*legs[active]+2]);
      V8 shift=config.poses[legs[active]];shift[2*legs[active]+1]=(legs[active]%2==0 ? .5:-.5);
      enter(SHIFT,shift);
    }
    const int leg=std::max(legs[active],0);
    elapsed+=dt;
    double duration=1;
    if(phase==ENTRY)duration=config.values[0];
    if(phase==SHIFT)duration=config.values[1];
    if(phase==LIFT)duration=config.values[2];
    if(phase==LOWER)duration=config.values[3];
    if(phase==RECENTER)duration=config.values[4];
    V8 desired=endpoint;
    if(phase!=ROTATE&&phase!=HOLD)for(int a=0;a<8;++a)desired[a]=start[a]+smooth(elapsed/duration)*(endpoint[a]-start[a]);
    for(int a=0;a<8;++a){
      const double limit=config.values[a%2==0 ? 12:13],accel=config.values[14];
      const double dv=std::clamp(36*(desired[a]-applied[a])-12*velocity[a],-accel,accel)*dt;
      velocity[a]=std::clamp(velocity[a]+dv,-limit,limit);
      applied[a]=std::clamp(applied[a]+velocity[a]*dt,Geometry::low[a],Geometry::high[a]);
    }
    double command_error=0,speed=0;
    for(int a=0;a<8;++a){command_error=std::max(command_error,std::abs(applied[a]-endpoint[a]));speed=std::max(speed,std::abs(velocity[a]));}
    const bool arrived=elapsed>=duration&&command_error<.015&&speed<.04;
    if(phase==ENTRY&&arrived)phase=HOLD;
    else if(phase==SHIFT&&arrived)enter(LIFT,config.poses[leg]);
    else if(phase==LOWER&&arrived)enter(RECENTER,Geometry::neutral);
    else if(phase==RECENTER&&arrived&&std::abs(q[3*leg+1])<.25&&std::abs(qd[3*leg+1])<.2){
      phase=HOLD;if(verified)completed|=1<<leg;
    }
    auto margins=Geometry::margins(q,gravity,leg);
    int blocked=0;
    if(phase!=ROTATE && !(phase==LIFT&&elapsed>=config.values[2]))blocked|=TRAJECTORY;
    if(margins[0]<=.010)blocked|=FLOOR;
    if(margins[1]<=.010)blocked|=WHEEL;
    if(margins[2]<=.005)blocked|=BODY;
    if(-gravity[2]<=std::cos(.12))blocked|=TILT;
    if(norm(angular)>=.3)blocked|=ANGULAR;
    if(phase==LIFT){gate_time=blocked==0 ? gate_time+dt:0;if(gate_time>=.2)phase=ROTATE;}
    std::array<double,4> wheel{};
    for(int j=0;j<4;++j)wheel[j]=std::clamp(2*wrap(hold[j]-q[3*j+2])-.35*qd[3*j+2],-.5,.5);
    const double error=wrap(target[leg]-q[3*leg+2]);
    if(phase==ROTATE&&blocked==0){
      const double reference_step=config.values[10]*dt;
      wheel_reference=wrap(wheel_reference+std::clamp(wrap(target[leg]-wheel_reference),-reference_step,reference_step));
      const double e=wrap(wheel_reference-q[3*leg+2]);
      // Integrate only near the final target. Conditional anti-windup and a
      // bounded integral contribution address sustained low-speed tracking error.
      const double candidate=std::clamp(integral+config.values[8]*e*dt,-config.values[9],config.values[9]);
      const double base=config.values[6]*e-config.values[7]*qd[3*leg+2];
      if(std::abs(error)<.3 && (std::abs(base+candidate)<=config.values[10] || e*(base+candidate)<0))integral=candidate;
      const double wanted=std::clamp(base+integral,-config.values[10],config.values[10]);
      wheel_speed+=std::clamp(wanted-wheel_speed,-config.values[11]*dt,config.values[11]*dt);
      wheel[leg]=wheel_speed;
      settled=std::abs(error)<.025&&std::abs(qd[3*leg+2])<.08 ? settled+dt:0;
      if(settled>=.5){verified=true;begin_lower(q);wheel[leg]=0;}
    }else if(is_up()){
      integral=settled=wheel_speed=0;wheel_reference=wrap(q[3*leg+2]);
      wheel[leg]=0; // Do not rotate through a failed clearance/stability gate.
    }
    output.position=applied;output.wheel=wheel;output.margins=margins;
    output.phase=phase;output.active=active;output.completed=completed;output.blocked=blocked;
    output.timeout=timeout;output.error=error;output.up_seconds=up_time;output.integral=integral;output.authority=true;
    return output;
  }
 private:
  V8 applied{},velocity{},start{},endpoint{};
  std::array<double,4> hold{},target{};
  double elapsed=0,up_time=0,gate_time=0,settled=0,integral=0,wheel_speed=0,wheel_reference=0;
  bool verified=false;
  void enter(Phase next,const V8 &end){phase=next;elapsed=0;start=applied;endpoint=end;}
  void begin_lower(const V12 &q){
    const int k=std::max(legs[active],0);hold[k]=wrap(q[3*k+2]);
    V8 landing=applied;const double sign=k%2==0 ? 1.:-1.;
    landing[2*k+1]=sign*std::clamp(sign*applied[2*k+1],0.,.5);
    integral=settled=wheel_speed=0;enter(LOWER,landing);
  }
};
}
