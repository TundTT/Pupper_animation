// Native and ARM64/QEMU smoke test of the exact standalone core. No hardware.
#include "controller.hpp"
#include <iostream>
#include <stdexcept>
int main(){
  keyframe_align::Controller c;keyframe_align::V12 q{1,0,.3,-1,0,-.4,1,0,.5,-1,0,-.6},qd{};
  const std::array<double,4> home{q[2],q[5],q[8],q[11]};c.reset(q,home);
  auto tick=[&](int command){auto o=c.step(.01,command,q,qd,{0,0,0},{0,0,-1});
    if(!o.authority)throw std::runtime_error("Lost authority");auto previous=q;
    for(int a=0;a<8;++a)q[keyframe_align::rows[a]]=o.position[a];
    for(int k=0;k<4;++k)q[3*k+2]+=o.wheel[k]*.01;
    for(int i=0;i<12;++i)qd[i]=(q[i]-previous[i])/.01;
  };
  for(int n=0;n<500;++n)tick(0);
  for(int request=1;request<=4;++request){
    for(int n=0;n<6000&&!(c.completed&(1<<keyframe_align::legs[request]));++n)tick(request);
    if(!(c.completed&(1<<keyframe_align::legs[request])))throw std::runtime_error("Missing completion");
  }
  for(int k=0;k<4;++k)if(std::abs(c.wrap(home[k]+std::acos(-1.)-q[3*k+2]))>=.035)throw std::runtime_error("Target retention");
  if(c.step(.01,0,q,qd,{0,0,0},{0,0,-1},true).authority)throw std::runtime_error("Stop");
  std::cout<<"PASS: portable core full sequence, final angle retention and stop\n";
}
