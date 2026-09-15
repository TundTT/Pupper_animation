#pragma once
#include "align_geometry.hpp"
#include <stdexcept>
namespace wheel_lift {
struct Sensors {
 std::array<double,12> q{},qd{};
 std::array<double,3> angular{},gravity{0,0,-1};
};
struct Core {
 enum Stage {idle,pose,lift,align,lower,done,fault};
 static constexpr const char* contract="leg-lift-wheel-position-v1";
 static constexpr std::array<int,8> proximal{0,1,3,4,6,7,9,10};
 static constexpr std::array<double,12> nominal{1,0,0,-1,0,0,1,0,0,-1,0,0};
 static constexpr std::array<double,12> scale{.5,1.6,0,.5,1.6,0,.5,1.6,0,.5,1.6,0};
 static constexpr std::array<double,12> lo{-1.12,-.32,-1000,-2.41,-3.04,-1000,-1.12,-.32,-1000,-2.41,-3.04,-1000};
 static constexpr std::array<double,12> hi{2.41,3.04,1000,1.12,.32,1000,2.41,3.04,1000,1.12,.32,1000};
 static constexpr std::array<double,12> kp{5,5,4,5,5,4,5,5,4,5,5,4};
 static constexpr std::array<double,12> kd{.25,.25,.15,.25,.25,.15,.25,.25,.15,.25,.25,.15};
 Stage stage=idle;int leg=0;
 double velocity=0,settled=0,elapsed=0,qualified=0;
 bool verified=false,rotating=false,supported=false,ready=false;
 std::array<double,4> home{},reference{},goal{},bottoms{};
 std::array<double,3> margins{};
 std::array<float,8> action{};
 std::array<float,27> observation{};
 std::array<double,12> command{},estimated_pd{};
 static bool finite(const Sensors& s){
  for(auto v:s.q)if(!std::isfinite(v))return false;
  for(auto v:s.qd)if(!std::isfinite(v))return false;
  for(auto v:s.angular)if(!std::isfinite(v))return false;
  for(auto v:s.gravity)if(!std::isfinite(v))return false;
  return true;
 }
 int actor_command()const{return (stage==lift||stage==align||stage==fault)?std::array<int,4>{2,1,3,4}[leg]:0;}
 void measure(const Sensors& s){
  if(!finite(s))throw std::invalid_argument("Invalid sensors");
  bottoms=WheelGeometry::bottoms(s.q,s.gravity);
  margins=AlignGeometry::margins(s.q,s.gravity,leg);
  double spin=0;for(double v:s.angular)spin+=v*v;
  bool stable=-s.gravity[2]>std::cos(.12)&&spin<.09;
  auto mm=std::minmax_element(bottoms.begin(),bottoms.end());
  supported=stable&&*mm.second-*mm.first<.006;
  for(double v:s.qd) supported=supported&&std::abs(v)<.1;
  // Same floor and pair bounds as the release, plus a conservative base-box guard.
  // These are encoder/IMU geometry estimates, not physical contact measurements.
  ready=AlignGeometry::gate(margins,s.gravity,s.angular);
 }
 void reset(const Sensors& s,const std::array<double,4>& startup){
  *this=Core{};home=startup;measure(s);
  if(!supported)throw std::invalid_argument("Entry requires stationary level support");
  for(int i:proximal)if(std::abs(s.q[i]-nominal[i])>.25)throw std::invalid_argument("Entry requires near-nominal proximal stand");
  for(int k=0;k<4;++k){
   if(!std::isfinite(home[k]))throw std::invalid_argument("Invalid calibrated home");
   reference[k]=s.q[3*k+2];
   double delta=std::fmod(home[k]+2*M_PI-reference[k],2*M_PI);if(delta<0)delta+=2*M_PI;delta-=M_PI;
   if(std::abs(delta+M_PI)<=1e-10)delta=M_PI;
   goal[k]=reference[k]+delta;
  }
  command=s.q;
 }
 bool press(){
  if(stage==idle)stage=pose;
  else if(stage==pose&&supported)stage=lift;
  else if(stage==lift&&ready){stage=align;elapsed=settled=qualified=0;verified=false;}
  else if(stage==align&&verified)stage=lower;
  else if(stage==lower&&supported){if(leg==3)stage=done;else{++leg;stage=lift;verified=false;}}
  else return false;
  return true;
 }
 void tick(double dt,const Sensors& s){
  if(!std::isfinite(dt)||dt<=0||dt>.02||!finite(s))throw std::invalid_argument("Invalid interval or sensors");
  if(stage!=align)return;
  const double q=s.q[3*leg+2],qd=s.qd[3*leg+2];
  if(verified){if(std::abs(q-goal[leg])>=.035||std::abs(qd)>=.08){verified=false;stage=fault;}return;}
  elapsed+=dt;
  if(elapsed>60){reference[leg]=q;velocity=0;stage=fault;return;}
  qualified=ready?qualified+dt:0;
  if(qualified<.2){if(rotating)reference[leg]=q;rotating=false;velocity=settled=0;return;}
  rotating=true;double error=goal[leg]-reference[leg];
  double desired=std::clamp(2*error,-.15,.15);
  velocity=std::clamp(desired,velocity-.3*dt,velocity+.3*dt);
  double advance=velocity*dt;
  if(std::abs(advance)>=std::abs(error)&&advance*error>=0){reference[leg]=goal[leg];velocity=0;}else reference[leg]+=advance;
  bool at=std::abs(q-goal[leg])<.025&&std::abs(qd)<.08&&std::abs(error)<.001;
  settled=at?settled+dt:0;
  if(settled>=.5){verified=true;velocity=0;reference[leg]=goal[leg];}
 }
 void observe(const Sensors& s){
  observation.fill(0);
  for(int i=0;i<3;++i){observation[i]=s.angular[i];observation[3+i]=s.gravity[i];}
  observation[6+actor_command()]=1;
  for(int i=0;i<8;++i){observation[11+i]=s.q[proximal[i]]-nominal[proximal[i]];observation[19+i]=action[i];}
 }
 void set_action(const float* a){for(int i=0;i<8;++i){if(!std::isfinite(a[i]))throw std::invalid_argument("Nonfinite actor output");action[i]=std::clamp(a[i],-1.f,1.f);}}
 void positions(const Sensors& s){
  for(int i=0;i<8;++i){int j=proximal[i];command[j]=std::clamp(nominal[j]+action[i]*scale[j],lo[j],hi[j]);}
  for(int k=0;k<4;++k)command[3*k+2]=reference[k];
  for(int i=0;i<12;++i){
   // Match the model's +/-3 Nm actuator force clamp while retaining position
   // actuation. Bound the PD estimate by moving the commanded target; do not
   // assume firmware effort_max clamps the motor's internal PD contribution.
   const double torque=kp[i]*(command[i]-s.q[i])-kd[i]*s.qd[i];
   if(!std::isfinite(torque))throw std::runtime_error("Nonfinite estimated PD");
   if(std::abs(torque)>3)command[i]=s.q[i]+(std::clamp(torque,-3.,3.)+kd[i]*s.qd[i])/kp[i];
   if(i%3!=2)command[i]=std::clamp(command[i],lo[i],hi[i]);
   estimated_pd[i]=kp[i]*(command[i]-s.q[i])-kd[i]*s.qd[i];
   if(!std::isfinite(estimated_pd[i])||std::abs(estimated_pd[i])>3.+1e-9)throw std::runtime_error("No position target within limits satisfies PD bound");

  }
 }
};
// Fractional scheduling: 520 Hz manager -> 50 Hz actor, at most one manager-tick jitter.
struct ActorClock {
 double elapsed=0;bool first=true;
 bool tick(double dt){elapsed+=dt;if(first){first=false;elapsed=0;return true;}if(elapsed+1e-12>=.02){elapsed-=.02;return true;}return false;}
};
}
