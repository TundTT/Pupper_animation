#include "neural_controller/leg_to_wheel/policy.hpp"
#include <iostream>
#include <sstream>
using namespace leg_to_wheel;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    require(argc==3,"Expected policy and recorded runtime trace");
    ManualSequencer seq;
    const std::array<int,8> expected{1,0,2,0,3,0,4,0};
    for(int step=1;step<=8;++step) {
      require(!seq.request(step+1),"Cannot skip a step");
      require(seq.request(step),"Next event accepted");
      require(seq.command()==expected[step-1],"FL lower FR lower BR lower BL lower");
      require(!seq.request(step),"Duplicate event rejected");
      require(seq.lower_requested()==(step%2==0),"Lower requested is distinct from contact");
    }
    require(!seq.request(9) && !seq.request(1),"No wrap or restart");
    Policy manual(argv[1]),above_floor(argv[1]),no_easing(argv[1]);
    std::ifstream input(argv[2]);std::string line;int count=0;bool eased=false;
    while(std::getline(input,line)) {
      std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
      Vec3 omega,gravity;Joints q;int command;
      for(auto& x:omega)row>>x;for(auto& x:gravity)row>>x;for(auto& x:q)row>>x;row>>command;
      require(bool(row),"Recorded input shape");
      auto a=manual.step_manual(omega,gravity,q,command);
      auto b=above_floor.step(omega,gravity,q,command,{.04,.04,.04,.04});
      auto c=no_easing.step(omega,gravity,q,command,{0,0,0,0});
      for(int i=0;i<12;++i) {
        require(std::abs(a.position_target[i]-b.position_target[i])<1e-10,"Time-only matches full-strength timed easing");
        require(a.raw_action[i]==c.raw_action[i],"Filter does not change raw action history");
        require(a.position_target[i]>=manual.lower[i] && a.position_target[i]<=manual.upper[i],"Trained target limits");
        if(std::abs(a.applied_action[i]-c.applied_action[i])>1e-5)eased=true;
      }
      ++count;
    }
    require(count==196 && eased,"Trace actually exercises lowering easing");
    manual.reset();auto initial=manual.step_manual({0,0,0},{0,0,-1},manual.home,1);
    for(int h=1;h<4;++h)for(int i=0;i<35;++i)
      require(manual.observation[i]==manual.observation[35*h+i],"Measured history reset");
    for(double x:initial.position_target)require(std::isfinite(x),"Finite first command");
    std::cout<<"PASS manual sequence and time-only easing on recorded policy inputs; no contact or floor data\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
