#include "neural_controller/wheel_align_motion.hpp"
#include "joy_utils/startup_home.hpp"
#include <iostream>
#include <stdexcept>
#include <iomanip>
using H=neural_controller::WheelAlignHybrid;
using M=neural_controller::WheelAlignMotion;
void check(bool x,const char* message){if(!x)throw std::runtime_error(message);}

int main(int argc,char**) {
  try {
    std::array<double,12> q{1,0,.3,-1,0,-.4,1,0,.5,-1,0,-.6},qd{};
    H h;h.reset(q);h.select_command(1,q);check(h.phase==H::IDLE,"uncalibrated command rejected");
    h.recalibrate_home(q);h.reset(q);auto home=h.home;
    q[2]+=17;h.reset(q);check(h.home==home,"startup home preserved");
    check(std::abs(h.hold[0]-H::wrap(q[2]))<1e-12,"hold refreshed after driving");
    check(std::abs(h.wheel_commands(q,qd)[0])<1e-12,"reentry does not chase stale angle");
    h.select_command(1,q);auto desired=h.applied_position;desired[3]=-1.6;
    auto result=h.limit_positions(desired,.02);check(std::abs(result[3]+.08)<1e-12,"legacy .08 rad hip cap");
    h.select_command(2,q);desired[3]=0;result=h.limit_positions(desired,.02);
    check(h.phase==H::LOWER&&h.active_command==1,"interrupt keeps old leg during descent");
    joy_utils::StartupHome capture;std::array<double,4> a{.1,.2,.3,.4},v{};
    for(int i=0;i<30;++i) capture.observe(a,v,i*.02);
    check(capture.captured,"startup captures stable home");auto fixed=capture.home;a[0]+=1;
    capture.observe(a,v,1.);check(capture.home==fixed,"startup home is immutable");
    joy_utils::StartupHome interrupted;interrupted.observe(a,v,0);interrupted.movement_requested=true;
    check(!interrupted.observe(a,v,1)&&!interrupted.captured,"movement before calibration refuses late capture");
    if(argc==1) {std::cout<<"PASS: startup/reentry/legacy limiter/interruption\n";return 0;}

    // Stream protocol for Python/C++ parity across realistic command sequences.
    double dt;int cmd;std::array<double,8> action{};M::Vec angular{},gravity{};M motion;
    bool first=true;std::cout<<std::setprecision(17);
    while(std::cin>>dt>>cmd) {
      for(auto&x:q)std::cin>>x;for(auto&x:qd)std::cin>>x;
      for(auto&x:angular)std::cin>>x;for(auto&x:gravity)std::cin>>x;for(auto&x:action)std::cin>>x;
      check(bool(std::cin),"truncated parity input");
      if(first){h=H{};h.motion_version=2;h.recalibrate_home(q);h.reset(q);motion.reset(q);first=false;}
      h.motion_lower_finished=motion.lower_finished(h);h.finish_step(q,qd);h.select_command(cmd,q);motion.prepare(h,dt);
      std::cout<<int(h.phase)<<' '<<h.active_command<<' '<<h.gate_steps<<' '<<h.settled_steps<<' '<<motion.progress<<' ';
      for(double x:motion.reference)std::cout<<x<<' ';
      const bool gate=motion.ready(h,q,angular,gravity);h.begin_step(q,gate,dt);
      auto wheels=h.wheel_commands(q,qd);motion.targets(h,action);
      for(int i=0;i<10;++i)motion.integrate(h,dt/10);
      for(double x:motion.applied)std::cout<<x<<' ';for(double x:motion.velocity)std::cout<<x<<' ';
      for(double x:wheels)std::cout<<x<<' ';for(bool x:h.completed)std::cout<<x<<' ';
      for(double x:M::margins(q,gravity,h.leg()))std::cout<<x<<' ';
      std::cout<<'\n';
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
