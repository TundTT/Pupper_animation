#pragma once
#include "lift_core.hpp"
#include "align_geometry.hpp"
namespace notebook_alignment_v4 {
// Optional extension: same learned ABI and proximal execution as LiftCore.
// Commands 0..4 retain their meaning; 5 requests rotation of the active leg.
struct LiftAlignCore : LiftCore {
 static constexpr const char* contract="quadmorph-notebook-lift-align-v1";
 bool alignment_enabled=false,rotation_requested=false,rotation_enabled=false,verified=false,stopping=false;
 int completed=0,failed=0,queued=-1;
 double qualified=0,settled=0,stop_elapsed=0,stop_settled=0;
 std::array<double,4> goal{},hub_velocity{};
 std::array<double,3> margins{};
 static double goal_near(double home,double current){
  double delta=std::remainder(home+std::acos(-1.)-current,2*std::acos(-1.));
  // At the exact half-turn tie, start +180deg is positive, never minus180.
  if(std::abs(std::abs(delta)-std::acos(-1.))<1e-10)delta=std::acos(-1.);
  return current+delta;
 }
 void reset(const LiftSensors& s,const std::array<double,4>& startup_home){
  LiftCore::reset(s,startup_home);rotation_requested=rotation_enabled=verified=stopping=false;
  completed=failed=0;queued=-1;qualified=settled=stop_elapsed=stop_settled=0;hub_velocity.fill(0);
  for(int k=0;k<4;++k)goal[k]=goal_near(home[k],hold[k]);
  margins=AlignGeometry::margins(s.q,s.gravity,request.leg);
 }
 bool gate(const LiftSensors& s){margins=AlignGeometry::margins(s.q,s.gravity,request.leg);return AlignGeometry::gate(margins,s.gravity,s.angular);}
 void capture_stop(const LiftSensors& s){
  // Safety reset, matching the notebook servo's one-shot gate-loss hold.
  // Do not accumulate position error behind a closed gate or recapture every tick.
  if(rotation_enabled&&!verified)hold[request.leg]=s.q[PositionTargets::hubs[request.leg]];
  rotation_enabled=false;hub_velocity.fill(0);qualified=settled=0;
 }
 void begin_stop(const LiftSensors& s,int next){
  queued=next;rotation_requested=false;
  if(!stopping){stopping=true;stop_elapsed=stop_settled=0;capture_stop(s);}
 }
 ObservationInput align_input(const LiftSensors& s)const{
  auto in=LiftCore::input(s);in.rotation_enabled=rotation_enabled;
  for(int k=0;k<4;++k)in.hub_velocity[k]=hub_velocity[k];
  return in;
 }
 void prepare(double dt,int event,const LiftSensors& s){
  if(!alignment_enabled){LiftCore::prepare(dt,event,s);return;}
  if(!finite(s)||!std::isfinite(dt)||dt<=0||dt>.1||event < -1||event>5)throw std::invalid_argument("Invalid align input");
  supported=support(s);stand_settle=supported?stand_settle+dt:0;
  const int old_phase=request.phase,old_leg=request.leg,k=old_leg;
  // A supported completion remains conditional on actual encoder settling.
  for(int j=0;j<4;++j)if(completed&(1<<j)){
   if(std::abs(goal[j]-s.q[PositionTargets::hubs[j]])>=.035||std::abs(s.qd[PositionTargets::hubs[j]])>=.08){completed&=~(1<<j);failed|=1<<j;}
  }
  if(verified&&std::abs(goal[k]-s.q[PositionTargets::hubs[k]])>.1)throw std::runtime_error("Aligned hub lost hold");
  if(event==5){if((request.phase==1||request.phase==2)&&!verified&&!stopping&&!(completed&(1<<k)))rotation_requested=true;event=-1;}
  if(event>0&&event<5&&(completed&(1<<ObservationHistory::leg_for_external_command(event))))event=-1;
  if((request.phase==1||request.phase==2)&&event>=0){
   int selected=ObservationHistory::leg_for_external_command(event);
   if(selected<0||selected!=k){begin_stop(s,event);event=-1;}
  }
  // Repeated requests during braking replace only the queue, not height/rate.
  if(stopping&&event>=0){queued=event;event=-1;}
  if((request.phase==1||request.phase==2)&&request.attempt+dt>=48){request.timed_out=true;begin_stop(s,0);}
  if(stopping){
   stop_elapsed+=dt;stop_settled=std::abs(s.qd[PositionTargets::hubs[k]])<.08?stop_settled+dt:0;
   if(stop_elapsed>2&&stop_settled<.2)throw std::runtime_error("Hub did not stop before lowering");
   if(stop_settled>=.2){request.lower();request.pending=queued>0?ObservationHistory::leg_for_external_command(queued):-1;stopping=false;queued=-1;}
  }
  // Suppress LiftRequest's timer-driven descent until the hub-stop check passes.
  double saved_attempt=request.attempt;
  if(stopping)request.attempt=0;
  request.advance(dt,event,supported&&(request.phase!=0||stand_settle>=.5));
  if(stopping)request.attempt=saved_attempt+dt;
  if(request.recovery_failed)throw std::runtime_error("Aligned supported recovery did not settle");
  if(request.phase==1&&(old_phase==0||request.leg!=old_leg)){
   rotation_requested=rotation_enabled=verified=stopping=false;qualified=settled=0;hub_velocity.fill(0);
  }
  if(old_phase==3&&request.phase==0){
   if(verified&&!request.timed_out&&std::abs(goal[k]-s.q[PositionTargets::hubs[k]])<.035&&std::abs(s.qd[PositionTargets::hubs[k]])<.08)completed|=1<<k;
   else failed|=1<<k;
   rotation_requested=false;
  }
  const bool good=gate(s)&&request.phase==2&&!stopping;
  qualified=good?qualified+dt:0;
  bool enabled=good&&qualified>=.2&&rotation_requested&&!verified;
  if(rotation_enabled&&!enabled)capture_stop(s);
  rotation_enabled=enabled;
  if(enabled){
   double desired=std::clamp(2*(goal[request.leg]-hold[request.leg]),-.5,.5);
   hub_velocity[request.leg]=std::clamp(desired,hub_velocity[request.leg]-1.2*dt,hub_velocity[request.leg]+1.2*dt);
   bool at=std::abs(goal[request.leg]-s.q[PositionTargets::hubs[request.leg]])<.025&&std::abs(s.qd[PositionTargets::hubs[request.leg]])<.08&&std::abs(goal[request.leg]-hold[request.leg])<.001;
   settled=at?settled+dt:0;
   if(settled>=.5){verified=true;rotation_requested=rotation_enabled=false;hub_velocity.fill(0);hold[request.leg]=goal[request.leg];}
  }else{hub_velocity.fill(0);if(!verified)settled=0;}
  // Reference velocity and permission are computed BEFORE this actor frame.
  history.update(align_input(s));
 }
 void execute(double dt,const LiftSensors& s){
  if(alignment_enabled){
   if(!finite(s)||!std::isfinite(dt)||dt<=0||dt>.01)throw std::invalid_argument("Invalid align command interval");
   // Recheck each520Hz tick. Safety gate loss overrides the last actor interval.
   if(rotation_enabled&&!gate(s))capture_stop(s);
   if(rotation_enabled){int k=request.leg;double next=hold[k]+hub_velocity[k]*dt;
    if((goal[k]-hold[k])*(goal[k]-next)<=0){next=goal[k];hub_velocity[k]=0;}
    hold[k]=next;
   }
  }
  LiftCore::execute(dt,s);
 }
};
}
