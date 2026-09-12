#include "neural_controller/inverted_triangle.hpp"
#include <json.hpp>
#include <fstream>
#include <iostream>
using namespace inverted_triangle;
void require(bool x,const char* why) { if(!x) throw std::runtime_error(why); }
Controller load(const char* path) {
  nlohmann::json j; std::ifstream f(path); f>>j; Controller c;
  c.initial=j.at("initial").get<Pose>(); c.kp=j.at("kp").get<Pose>(); c.kd=j.at("kd").get<Pose>();
  for(const auto& s:j.at("segments")) c.segments.push_back({s.at("stage"),s.at("phase"),s.at("steps"),s.at("target").get<Pose>()});
  c.validate(); return c;
}
#ifndef TRIANGLE_BRIDGE
int main(int argc,char** argv) {
  try {
    require(argc==2,"Expected plan JSON"); auto c=load(argv[1]); Pose q=c.initial,v{},mapping{};
    mapping[2]=10; mapping[5]=-20; mapping[8]=30; mapping[11]=-40;
    for(int i=0;i<12;++i) q[i]+=mapping[i];
    c.reset(q,mapping);
    auto tick=[&](int request=0,bool fresh=false) {
      c.step(1./hz,q,v,0,request,fresh);
      for(int i=0;i<12;++i) { double next=c.command[i]+mapping[i]; v[i]=(next-q[i])*hz; q[i]=next; }
      require(c.state!=FAULT,"Nominal fake tracking must not fault");
    };
    for(int i=0;i<1041;++i) tick();
    require(c.state==READY,"Activation waits for explicit command");
    tick(2,true); require(c.state==READY,"Cannot skip first leg");
    int running_steps=0;
    for(int leg=1;leg<=4;++leg) {
      tick(leg,true); ++running_steps;
      for(int n=0;c.state==RUNNING && n<30000;++n) {
        tick(leg+1,n==2 && leg<4); ++running_steps; // Early next-leg command is discarded.
      }
      require(c.completed==leg,"One request completes exactly one leg");
      const Pose held=c.command;
      for(int n=0;n<520;++n) tick();
      require(c.command==held,"Pause preserves planted endpoint and encoder winding");
      tick(leg,true); require(c.completed==leg && c.state!=RUNNING,"Duplicate request cannot repeat a flip");
    }
    require(c.state==DONE && running_steps==67278,"Full trajectory step parity");
    require(std::abs(c.command[11]-7.283185307179793)<1e-10,"Rear-left extra revolution preserved");
    auto reset=[&]{q=c.initial; v.fill(0); mapping.fill(0);c.reset(q,mapping);};
    reset(); c.step(.011,q,v,0); require(c.fault==TIMING,"Timing watchdog");
    c.step(1./hz,q,v,0,1,true);require(c.state==FAULT && c.gain==0,"Fault cannot restart from command");
    reset(); c.step(1./hz,q,v,.15); require(c.fault==TILT,"Eight degree tilt guard");
    reset(); q[2]+=.151; c.step(1./hz,q,v,0);require(c.fault==TRACKING,"Hub tracking guard");
    reset(); v[1]=2.01;c.step(1./hz,q,v,0);require(c.fault==SPEED,"Speed guard");
    reset(); q[0]=NAN;c.step(1./hz,q,v,0);require(c.fault==SENSORS,"Nonfinite encoder guard");
    reset();c.step(1./hz,q,v,0,-1,true);require(c.fault==OPERATOR && c.gain==0,"Operator abort releases gains");
    reset();q[0]+=.04;bool rejected=false;try{c.reset(q,mapping);}catch(const std::exception&){rejected=true;}
    require(rejected,"No approach from incorrect initial pose");
    reset();mapping[0]=.01;rejected=false;try{c.reset(q,mapping);}catch(const std::exception&){rejected=true;}
    require(rejected,"Never invent proximal encoder offsets");
    std::cout<<"PASS: 67278 target steps, four stage requests, pauses, unwrapped mapping, guards and latched stops\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#else
// Test-only bridge: feed actual MuJoCo feedback through the same C++ core.
// Starts after the externally supported arming ramp; never links into the robot.
extern "C" {
void* triangle_create(const char* path) {
  try { auto* c=new Controller(load(path)); c->reset(c->initial,Pose{}); c->state=READY; c->gain=1.; return c; }
  catch(...) { return nullptr; }
}
void triangle_destroy(void* p) { delete static_cast<Controller*>(p); }
void triangle_tick(void* p,const double* q,const double* v,double tilt,int request,double dt,double* output) {
  auto& c=*static_cast<Controller*>(p);Pose qp{},vp{};std::copy_n(q,12,qp.begin());std::copy_n(v,12,vp.begin());
  c.step(dt,qp,vp,tilt,request,request!=0);std::copy(c.command.begin(),c.command.end(),output);
  output[12]=c.state;output[13]=c.completed;output[14]=c.fault;output[15]=c.segment;
}
}
#endif
