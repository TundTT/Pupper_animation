#include "neural_controller/measured_roll.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
using namespace measured_roll;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::vector<double> row(std::istream& f) {
  std::string line,token;std::getline(f,line);std::stringstream s(line);std::vector<double> out;
  while(std::getline(s,token,','))out.push_back(std::stod(token));return out;
}
int main(int argc,char** argv) {
 try {
  // The approved policy home is shared by alignment and both roll endpoints.
  for(int leg=0;leg<4;++leg)for(int joint=0;joint<2;++joint) {
    const auto expected=neural_controller::policy_home::upper[2*leg+joint];
    check(point_up[3*leg+joint]==expected && neutral[3*leg+joint]==expected,
          "Roll must inherit approved upper home, independently of hub calibration");
  }
  if(argc==2) {
    std::ifstream file(argv[1]);check(bool(file),"Missing recorded fixture");auto initial=row(file);
    check(initial.size()==24,"Invalid initial fixture");Pose q{},qd{},previous{};
    for(int i=0;i<12;++i){q[i]=initial[i];previous[i]=initial[12+i];}
    Motion motion;motion.reset(q,previous);triangle_roll::Support support;support.set_goal(motion.goal);
    int count=0;double peak=0;
    while(file.peek()!=EOF) {
      auto v=row(file);check(v.size()==51,"Invalid replay row");
      for(int i=0;i<12;++i){q[i]=v[i];qd[i]=v[12+i];}
      V3 gravity(v[24],v[25],v[26]);
      if(motion.phase==SETTLE && count%10==0)support.update(q,qd,motion.command,gravity,10./hz);
      motion.step(q,qd,1./hz,support.offset);
      for(int i=0;i<12;++i) {
        peak=std::max(peak,std::abs(motion.command[i]-v[27+i]));
        double kd=triangle_roll::base_kd[i]*(motion.phase==ENTRY || motion.phase==ROLL?2.:1.);
        check(kd==v[39+i],"Damping/phase differs from saved run");
      }
      check(motion.phase!=FAILED,"Saved motion failed in C++");++count;
    }
    std::cout<<"Recorded replay: "<<count<<" steps, peak command error "<<peak<<" rad\n";
    check(count==24051 && motion.phase==FINISHED && peak<1e-8,"C++ differs from passing recorded controller");
  }
  // Real executor envelope, with ideal feedback and a non-exact entry pose.
  for(bool suspended:{false,true}) {
    measured_roll::Controller c;c.stand_only=suspended;Pose q=point_up,qd{},mapping{};
    q[0]+=.04;q[1]=.035;q[4]=-.035;
    if(suspended)for(int leg=0;leg<4;++leg)q[3*leg+1]=leg%2==0?-.179972:.179973;
    for(int h:{2,5,8,11}){mapping[h]=6*pi;q[h]+=mapping[h];}
    c.reset(q,mapping);Pose held=c.command;
    for(int k=0;k<1041;++k)c.step(1./hz,q,qd,{0,0,-1},1,k==0);
    check(c.state==READY && c.command==held,"Activation must hold current pose; early START discarded");
    c.step(1./hz,q,qd,{0,0,-1},1,true);
    Pose last=c.command,last_velocity{};
    while(c.state==RUNNING && c.run_steps<27000) {
      for(int i=0;i<12;++i){double next=c.command[i]+mapping[i];qd[i]=(next-q[i])*hz;q[i]=next;}
      c.step(1./hz,q,qd,{0,0,-1});
      for(int i=0;i<12;++i) {
        double v=(c.command[i]-last[i])*hz;
        check(std::abs(v)<=speed[i]+1e-9,"Command speed limit");
        check(std::abs(v-last_velocity[i])*hz<=accel[i]+1e-7,"Command acceleration limit");
        last_velocity[i]=v;
      }
      last=c.command;
    }
    check(c.state==DONE && c.completed==1,"Executor should end in a hold");
    if(suspended)check(c.support.elapsed==0,"Suspended mode must never infer ground loads");
    for(int leg=0;leg<4;++leg) {
      int h=3*leg+2;
      check(std::abs(c.motion.goal[h]-held[h]-(leg%2==0?-pi:pi))<1e-10,"Forward half turn, never extra winding");
    }
    held=c.command;qd.fill(0);for(int i=0;i<12;++i)q[i]=held[i]+mapping[i];
    for(int k=0;k<50;++k)c.step(1./hz,q,qd,{0,0,-1},1,true);
    check(c.state==DONE && c.command==held,"Done cannot repeat or start walking");
    c.step(1./hz,q,qd,{0,0,-1},-1,true);check(c.state==FAULT && c.gain==0,"Stop releases torque");
    c.step(1./hz,q,qd,{0,0,-1},1,true);check(c.state==FAULT,"Fault is latched");
  }
  measured_roll::Controller c;Pose q=point_up,qd{};
  c.reset(q,Pose{});c.step(.02,q,qd,{0,0,-1});check(c.fault==TIMING,"Timing guard");
  c.reset(q,Pose{});c.step(1./hz,q,qd,{0,.2,-std::sqrt(.96)});check(c.fault==TILT,"Tilt guard");
  c.reset(q,Pose{});qd[2]=2.1;c.step(1./hz,q,qd,{0,0,-1});check(c.fault==SPEED,"Speed guard");qd.fill(0);
  c.hub_tracking_error_limit=pi/6.;c.validate();q=point_up;c.reset(q,Pose{});
  q[2]+=.51;c.step(1./hz,q,qd,{0,0,-1});check(c.state!=FAULT,"30-degree tracking limit permits smaller error during gain ramp");
  check(c.kp==triangle_roll::base_kp,"Retry leaves motor gains unchanged");
  q=point_up;c.reset(q,Pose{});q[2]+=pi/6.+.001;c.step(1./hz,q,qd,{0,0,-1});check(c.fault==TRACKING,"Retry still bounds hub error at 30 degrees");
  q=point_up;c.reset(q,Pose{});q[1]+=.251;c.step(1./hz,q,qd,{0,0,-1});check(c.fault==TRACKING,"Upper tracking guard unchanged");
  q=point_up;c.reset(q,Pose{});c.state=READY;c.gain=1;q[2]+=.4;
  c.step(1./hz,q,qd,{0,0,-1});check(c.fault==TORQUE,"Torque guard remains active below enlarged tracking limit");
  c.hub_tracking_error_limit=.7;bool tracking_rejected=false;try{c.validate();}catch(...){tracking_rejected=true;}check(tracking_rejected,"Reject tolerance above operator-selected 30 degrees");
  c.hub_tracking_error_limit=.15;q=point_up;
  c.reset(q,Pose{});q[2]+=.16;c.step(1./hz,q,qd,{0,0,-1});check(c.fault==TRACKING,"Tracking guard");
  q=point_up;c.reset(q,Pose{});c.step(1./hz,q,qd,{0,0,-1},2,true);check(c.fault==COMMAND,"Unknown commands rejected");
  q[0]+=.3;bool rejected=false;try{c.reset(q,Pose{});}catch(...){rejected=true;}check(rejected,"Unknown proximal pose rejected");
  std::cout<<"PASS: measured entry, continuous winding, acceleration bounds, ground/stand hold and fault guards\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
