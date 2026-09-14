#pragma once
#include "observation.hpp"
namespace notebook_alignment_v4 {
// Request-only state: the actor, never this helper, chooses proximal targets.
struct LiftRequest {
 int leg=0,phase=0,pending=-1;
 double elapsed=0,attempt=0,height=0,rate=0,lower_height=0,lower_rate=0,support_time=0;
 bool timed_out=false,recovery_failed=false;
 static constexpr double target=.008;
 void reset(){*this=LiftRequest{};}
 void start(int selected){leg=selected;phase=1;elapsed=attempt=height=rate=0;pending=-1;timed_out=false;}
 void lower(){if(phase==3)return;lower_height=height;lower_rate=rate;phase=3;elapsed=support_time=0;}
 void advance(double dt,int new_command,bool supported) {
  if(!std::isfinite(dt)||dt<=0||dt>.1)throw std::invalid_argument("Invalid actor interval");
  if(new_command < -1 || new_command>4)throw std::invalid_argument("Invalid lift request");
  // -1 means no new message. Repeated stand never restarts a moving descent.
  if(new_command>=0) {
   int selected=ObservationHistory::leg_for_external_command(new_command);
   if(selected<0){pending=-1;if(phase==1||phase==2)lower();}
   else if(phase==0&&supported){start(selected);}
   else if(phase==0){pending=selected;}
   else if(selected!=leg){pending=selected;lower();}
  }
  if(phase==0) {
   height=rate=0;
   if(pending>=0&&supported&&!timed_out)start(pending);
   return;
  }
  elapsed+=dt;attempt+=dt;
  if(attempt>=48&&(phase==1||phase==2)){timed_out=true;pending=-1;lower();}
  double u=std::clamp(elapsed/4.,0.,1.);
  if(phase==1){height=target*u*u*(3-2*u);rate=target*6*u*(1-u)/4.;if(u>=1){phase=2;elapsed=0;}}
  else if(phase==2){height=target;rate=0;}
  else {
   height=lower_height*(2*u*u*u-3*u*u+1)+4*lower_rate*(u*u*u-2*u*u+u);
   rate=lower_height*(6*u*u-6*u)/4+lower_rate*(3*u*u-4*u+1);
   support_time=(elapsed>=4&&supported)?support_time+dt:0;
   if(support_time>=.5){phase=0;height=rate=elapsed=0;}
   if(elapsed>14)recovery_failed=true; // bounded4s ramp plus10s supported recovery
  }
 }
 int actor_phase()const{return phase==3&&elapsed>=4 ? 0:phase;}
};
}
