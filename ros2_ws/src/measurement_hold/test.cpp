#include "core.hpp"
#include <iostream>
#include <limits>
using namespace measurement_hold;
void require(bool x){if(!x)throw std::runtime_error("hold test failed");}
int main(){
  Pose q{.9,-.18,2.7,-.9,.18,-3.5,.9,-.18,8.,-.9,.18,-9.},v{};
  Core c;c.reset(q,v);require(c.gain==0);
  double previous=0;
  for(int i=0;i<1561;++i){c.step(1./520,q,v,true);require(c.target==q&&c.gain>=previous&&c.gain-previous<.001);previous=c.gain;}
  require(c.gain==1&&!c.stopped);
  c.step(1./520,q,v,false);require(c.stopped&&c.gain==0);
  c.step(1./520,q,v,true);require(c.stopped&&c.gain==0);
  c.reset(q,v);Pose bad=q;bad[3]+=.26;c.step(1./520,bad,v,true);require(c.stopped);
  c.reset(q,v);c.step(.02,q,v,true);require(c.stopped);
  c.reset(q,v);bad=q;bad[1]=std::numeric_limits<double>::quiet_NaN();c.step(1./520,bad,v,true);require(c.stopped);
  bool rejected=false;v[0]=.2;try{c.reset(q,v);}catch(...){rejected=true;}require(rejected);
  std::cout<<"PASS: fixed measured targets, 3s gain ramp, stop latch, error/speed/timing/nonfinite checks\n";
}
