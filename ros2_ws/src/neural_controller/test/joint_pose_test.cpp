#include "neural_controller/joint_pose.hpp"
#include <iostream>
using namespace joint_pose;
void check(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(){try{
  Core fault_latch;fault_latch.stop(INPUT);fault_latch.stop(STOP);
  check(fault_latch.fault==INPUT,"Preserve the first fault rather than overwriting it with stop latch");
  // Actual displaced encoder snapshot; only the hub winding is chosen before reset.
  Pose q{1.1202386,-.3264583,-1.1336429,-.7586025,.5755628,1.2697495,
         1.2747367,-.1193174,-1.1599653,-.6308083,-.0576865,2.4797905};
  Pose goal{1,0,1.9223568,-1,0,4.4072877,1,0,1.9513495,-1,0,4.0929519},v{};
  Core c;c.reset(q,v,goal);Pose oldref=c.reference,oldvelocity{};double peak=0;
  for(int k=0;k<40000&&c.state!=HOLD&&c.state!=FAULT;++k){
    c.step(1./520,q,v);
    for(int i=0;i<12;++i){
      check(c.reference[i]>=low[i]&&c.reference[i]<=high[i],"Reference left operational bounds");
      double speed=(c.reference[i]-oldref[i])*520;
      check(std::abs(speed)<(i%3==2?.501:.301),"Reference speed");
      check(std::abs(speed-oldvelocity[i])*520<(i%3==2?1.21:.81),"Reference acceleration");
      oldvelocity[i]=speed;
      double torque=c.kp[i]*(c.reference[i]-q[i])-c.kd[i]*v[i];peak=std::max(peak,std::abs(torque));
      check(std::abs(torque)<=.60000001,"Torque cap");
      // Damped approximate plant exercises recovery; not a physical validation.
      v[i]+=(torque-.25*v[i])/.04/520;q[i]+=v[i]/520;
    }
    oldref=c.reference;
  }
  check(c.state==HOLD,"Displaced-pose recovery did not finish");
  check(c.total<40,"Unexpected duration");
  for(int i=0;i<12;++i)check(std::abs(q[i]-goal[i])<.06,"Final tracking");
  c.step(.03,q,v);check(c.state==FAULT&&c.kp==Pose{},"Timing stop");
  c.reset(q,Pose{},goal);q[0]=NAN;c.step(.002,q,v);check(c.state==FAULT,"NaN stop");
  q=goal;c.reset(q,Pose{},goal);v.fill(0);v[0]=2;c.step(.002,q,v);check(c.state==FAULT,"Speed stop");
  q=goal;q[4]=high[4]+.36;bool rejected=false;try{c.reset(q,Pose{},goal);}catch(...){rejected=true;}check(rejected,"Recovery allowance enforced");
  q=goal;goal[2]+=6.28;rejected=false;try{c.reset(q,Pose{},goal);}catch(...){rejected=true;}check(rejected,"Full-turn ambiguity rejected");
  std::cout<<"PASS: actual displaced start, bounded inward recovery, staged trajectory, speed/acceleration, peak torque "<<peak<<", stop guards\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
