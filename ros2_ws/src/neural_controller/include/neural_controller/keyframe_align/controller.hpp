#pragma once
#include "geometry.hpp"
#include <stdexcept>

namespace keyframe_align {
using V8 = std::array<double,8>;
using V12 = std::array<double,12>;
using V3 = std::array<double,3>;
inline constexpr std::array<int,8> rows{0,1,3,4,6,7,9,10};
inline constexpr std::array<int,5> legs{-1,1,0,2,3};
enum Phase { ENTRY, SHIFT, LIFT, ROTATE, LOWER, RECENTER, HOLD, STOPPED };
enum Block { TRAJECTORY=1, FLOOR=2, WHEEL=4, BODY=8, TILT=16, ANGULAR=32 };
struct Config {
  double rotation_floor_clearance_m = .010;
  double wheel_position_kp=2., wheel_position_kd=.15;
  double alignment_angle_tolerance_rad=.025, landing_angle_tolerance_rad=.035;
  double alignment_speed_tolerance_rad_s=.08, alignment_settle_seconds=.5;
  double hold_error_limit_rad=.10;
  double wheel_integral_ki=.5, wheel_integral_limit_nm=.10, wheel_integral_window_rad=.10;
  // Keep timing/trajectory indices stable; slots 6..9 are retired outer-loop gains.
  std::array<double,15> values{2.,1.5,1.5,3.,1.5,48.,0.,0.,0.,0.,.5,1.2,.45,.65,2.};
  std::array<V8,4> poses{{
    {{1.16890458,1.0021502200000001,-1.2625986899999999,0.28754656000000001,1.0167859299999999,-0.16507621,-0.60319067999999998,-0.19620607000000001}},
    {{1.2625986899999999,-0.28754656000000001,-1.16890458,-1.0021502200000001,0.60319067999999998,0.19620607000000001,-1.0167859299999999,0.16507621}},
    {{1,0,-1,0,1,1.2,-1,0}},
    {{1,0,-1,0,1,0,-1,-1.2}},
  }};
  void validate() const {
    if(!std::isfinite(wheel_integral_ki)||wheel_integral_ki<0||wheel_integral_ki>5||
       !std::isfinite(wheel_integral_limit_nm)||wheel_integral_limit_nm<=0||wheel_integral_limit_nm>.25||
       !std::isfinite(wheel_integral_window_rad)||wheel_integral_window_rad<alignment_angle_tolerance_rad||
       wheel_integral_window_rad>hold_error_limit_rad)
      throw std::invalid_argument("Invalid bounded wheel integral settings");
    if(!std::isfinite(rotation_floor_clearance_m)||rotation_floor_clearance_m<=0||rotation_floor_clearance_m>.1)
      throw std::invalid_argument("Rotation floor clearance must be finite and in (0, 0.1] metres");
    for(double x:{wheel_position_kp,wheel_position_kd,alignment_angle_tolerance_rad,
                  landing_angle_tolerance_rad,alignment_speed_tolerance_rad_s,
                  alignment_settle_seconds,hold_error_limit_rad})
      if(!std::isfinite(x)||x<=0)throw std::invalid_argument("Positive finite wheel position settings required");
    if(wheel_position_kp>10||wheel_position_kd>1||alignment_angle_tolerance_rad>hold_error_limit_rad||
       landing_angle_tolerance_rad>hold_error_limit_rad||hold_error_limit_rad>.3)
      throw std::invalid_argument("Wheel position settings outside hardware/hold limits");
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
  std::array<double,4> wheel{}, wheel_position{}, wheel_effort{};
  V3 margins{};
  int phase=ENTRY,active=0,completed=0,blocked=TRAJECTORY,timeout=-1,failed=0;
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
    phase=ENTRY;active=completed=0;timeout=-1;elapsed=up_time=gate_time=settled=0;rotating=false;
    verified=false;failed=0;landing_settled=0;velocity.fill(0);torque_bias.fill(0);
    for(int a=0;a<8;++a) applied[a]=q[rows[a]];
    for(int k=0;k<4;++k){hold[k]=q[3*k+2];home[k]=calibrated_home[k];target[k]=q[3*k+2]+wrap(home[k]+std::acos(-1.)-q[3*k+2]);}
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
    if(phase==STOPPED){output.authority=false;output.phase=STOPPED;output.wheel.fill(0);torque_bias.fill(0);output.wheel_effort.fill(0);output.integral=0;return output;}
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
      active=requested;up_time=0;verified=false;gate_time=settled=0;rotating=false;
      failed&=~(1<<legs[active]);
      const int k=legs[active];torque_bias[k]=0;target[k]=q[3*k+2]+wrap(home[k]+std::acos(-1.)-q[3*k+2]);
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
      const bool aligned=std::abs(wrap(target[leg]-q[3*leg+2]))<config.landing_angle_tolerance_rad&&std::abs(qd[3*leg+2])<config.alignment_speed_tolerance_rad_s;
      landing_settled=aligned ? landing_settled+dt:0;
      if(!verified || landing_settled>=config.alignment_settle_seconds || elapsed>=config.values[4]+4){
        phase=HOLD;
        if(verified && landing_settled>=config.alignment_settle_seconds)completed|=1<<leg;
        else if(verified){failed|=1<<leg;timeout=active;torque_bias[leg]=0;}
      }
    }
    auto margins=Geometry::margins(q,gravity,leg);
    int blocked=0;
    if(phase!=ROTATE && !(phase==LIFT&&elapsed>=config.values[2]))blocked|=TRAJECTORY;
    if(margins[0]<=config.rotation_floor_clearance_m)blocked|=FLOOR;
    if(margins[1]<=.010)blocked|=WHEEL;
    if(margins[2]<=.005)blocked|=BODY;
    if(-gravity[2]<=std::cos(.12))blocked|=TILT;
    if(norm(angular)>=.3)blocked|=ANGULAR;
    if(phase==LIFT){gate_time=blocked==0 ? gate_time+dt:0;if(gate_time>=.2)phase=ROTATE;}
    std::array<double,4> wheel_position=hold;
    for(int j=0;j<4;++j)if(completed&(1<<j)){
      const double e=target[j]-q[3*j+2];
      if(e*torque_bias[j]<0)torque_bias[j]=0;
      wheel_position[j]=target[j];
      if(std::abs(e)>=config.hold_error_limit_rad){
        completed&=~(1<<j);failed|=1<<j;torque_bias[j]=0;hold[j]=q[3*j+2];wheel_position[j]=hold[j];
      }
    }
    // Preserve the aligned absolute angle through lowering. A large disturbance
    // releases that target instead of attempting another rotation on the ground.
    if(verified && (phase==LOWER||phase==RECENTER||phase==HOLD)) {
      if((hold[leg]-q[3*leg+2])*torque_bias[leg]<0)torque_bias[leg]=0;
      if(std::abs(hold[leg]-q[3*leg+2])>=config.hold_error_limit_rad){
        failed|=1<<leg;verified=false;timeout=active;torque_bias[leg]=0;hold[leg]=q[3*leg+2];
      }
      wheel_position[leg]=hold[leg];
    }
    const double error=wrap(target[leg]-q[3*leg+2]);
    if(phase==ROTATE&&blocked==0){
      if(!rotating){
        rotation_start=q[3*leg+2];
        target[leg]=rotation_start+wrap(home[leg]+std::acos(-1.)-rotation_start);
        rotation_delta=target[leg]-rotation_start;rotation_elapsed=0;
        // Quintic smoothstep has peak slope 1.875 and peak curvature 10/sqrt(3).
        rotation_duration=std::max({dt,1.875*std::abs(rotation_delta)/config.values[10],
          std::sqrt((10/std::sqrt(3.))*std::abs(rotation_delta)/config.values[11])});
        rotating=true;
      }
      rotation_elapsed+=dt;
      wheel_position[leg]=rotation_start+smooth(rotation_elapsed/rotation_duration)*rotation_delta;
      const double final_error=target[leg]-q[3*leg+2];
      // A small torque bias supplements the motor position PD. Accumulate only
      // after the angle ramp, near target, and at low measured speed. Discard
      // opposing bias immediately after crossing the target; clamp at all times.
      if(final_error*torque_bias[leg]<0 || std::abs(final_error)>config.wheel_integral_window_rad)
        torque_bias[leg]=0;
      if(rotation_elapsed>=rotation_duration&&std::abs(final_error)<=config.wheel_integral_window_rad&&
         std::abs(final_error)>config.alignment_angle_tolerance_rad&&
         std::abs(qd[3*leg+2])<config.alignment_speed_tolerance_rad_s)
        torque_bias[leg]=std::clamp(torque_bias[leg]+config.wheel_integral_ki*final_error*dt,
          -config.wheel_integral_limit_nm,config.wheel_integral_limit_nm);
      const bool at_target=rotation_elapsed>=rotation_duration&&
        std::abs(target[leg]-q[3*leg+2])<config.alignment_angle_tolerance_rad&&
        std::abs(qd[3*leg+2])<config.alignment_speed_tolerance_rad_s;
      settled=at_target ? settled+dt:0;
      if(settled>=config.alignment_settle_seconds){verified=true;begin_lower(q);wheel_position[leg]=hold[leg];}
    }else if(is_up()){
      settled=0;rotating=false;torque_bias[leg]=0;
      // No accumulated position demand behind a closed rotation gate.
      hold[leg]=q[3*leg+2];wheel_position[leg]=hold[leg];
    }
    output.position=applied;output.wheel.fill(0);output.wheel_position=wheel_position;output.wheel_effort=torque_bias;output.margins=margins;
    output.phase=phase;output.active=active;output.completed=completed;output.blocked=blocked;
    output.timeout=timeout;output.failed=failed;output.error=error;output.up_seconds=up_time;output.integral=torque_bias[leg];output.authority=true;
    return output;
  }
 private:
  V8 applied{},velocity{},start{},endpoint{};
  std::array<double,4> hold{},target{},home{},torque_bias{};
  double elapsed=0,up_time=0,gate_time=0,settled=0;
  double rotation_start=0,rotation_delta=0,rotation_elapsed=0,rotation_duration=1;
  bool rotating=false;
  bool verified=false;
  int failed=0;
  double landing_settled=0;
  void enter(Phase next,const V8 &end){phase=next;elapsed=0;start=applied;endpoint=end;}
  void begin_lower(const V12 &q){
    const int k=std::max(legs[active],0);hold[k]=verified ? target[k]:q[3*k+2];landing_settled=0;
    if(!verified)torque_bias[k]=0; // Cancel/timeout must not carry a nudging torque into descent.
    V8 landing=applied;const double sign=k%2==0 ? 1.:-1.;
    landing[2*k+1]=sign*std::clamp(sign*applied[2*k+1],0.,.5);
    settled=0;rotating=false;enter(LOWER,landing);
  }
};
}
