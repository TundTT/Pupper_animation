#include "neural_controller/leg_to_wheel/policy.hpp"
#include <iostream>
#include <sstream>
#include <vector>
using namespace leg_to_wheel;
static void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
static std::vector<double> numbers(std::string line) {
  std::replace(line.begin(),line.end(),',',' ');std::istringstream stream(line);std::vector<double> values;double x;
  while(stream>>x)values.push_back(x);return values;
}
int main(int argc,char** argv) {
  try {
    require(argc==4,"usage: TEST POLICY NETWORK_CSV RUNTIME_CSV");
    std::ifstream input(argv[1]);auto net=RTNeural::json_parser::parseJson<float>(input,false);
    require(net->getInSize()==140 && net->getOutSize()==12,"network dimensions");
    std::ifstream fixtures(argv[2]);std::string line;int count=0;double network_error=0.,runtime_error=0.;
    while(std::getline(fixtures,line)) {
      auto row=numbers(line);require(row.size()==152,"network fixture width");std::array<float,140> obs;
      std::copy_n(row.begin(),140,obs.begin());net->forward(obs.data());
      for(int i=0;i<12;++i) {double e=std::abs(net->getOutputs()[i]-row[140+i]);require(std::isfinite(e) && e<3e-5,"network parity");network_error=std::max(network_error,e);}
      ++count;
    }
    require(count==128,"network fixture count");
    Policy policy(argv[1]);std::ifstream runtime(argv[3]);count=0;
    auto compare=[&](double actual,double expected) {double e=std::abs(actual-expected);require(std::isfinite(e) && e<3e-5,"runtime parity");runtime_error=std::max(runtime_error,e);};
    while(std::getline(runtime,line)) {
      auto row=numbers(line);require(row.size()==199,"runtime fixture width");Vec3 omega,gravity;Joints q;std::array<double,4> clearance;
      std::copy_n(row.begin(),3,omega.begin());std::copy_n(row.begin()+3,3,gravity.begin());std::copy_n(row.begin()+6,12,q.begin());std::copy_n(row.begin()+19,4,clearance.begin());
      auto out=policy.step(omega,gravity,q,int(row[18]),clearance);
      for(int i=0;i<140;++i)compare(policy.observation[i],row[23+i]);
      for(int i=0;i<12;++i) {compare(out.raw_action[i],row[163+i]);compare(out.applied_action[i],row[175+i]);compare(out.position_target[i],row[187+i]);}
      ++count;
    }
    require(count==196,"runtime fixture count");
    // Reset removes command/history/target state before another activation.
    policy.reset();auto out=policy.step({0,0,0},{0,0,-1},policy.home,0,{.04,.04,.04,.04});
    for(int h=1;h<4;++h)for(int i=0;i<35;++i)require(policy.observation[i]==policy.observation[35*h+i],"history seeding");
    bool rejected=false;try {policy.step({0,0,0},{0,0,-1},policy.home,5,{0,0,0,0});}catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid command rejection");
    rejected=false;try {policy.step({0,0,0},{0,0,-1},policy.home,0,{0,0,0,0},0.);}catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid clock rejection");
    Sequencer fsm;
    for(int command=1;command<=4;++command) {
      require(fsm.request(command),"leg request");require(!fsm.request(command),"overlapping request");
      for(int i=0;i<30;++i)fsm.update(.02,.01,false,true);
      require(fsm.command==command && fsm.phase==Sequencer::Phase::lifting,"real clearance gate");
      for(int i=0;i<3100;++i)fsm.update(.02,.03,false);
      require(fsm.command==command && fsm.phase==Sequencer::Phase::heating,"indefinite operator hold");
      fsm.update(0.,.03,false,true);require(fsm.command==0,"explicit heating confirmation");
      for(int i=0;i<10;++i)fsm.update(.02,0.,true);
      require(fsm.phase==Sequencer::Phase::lowering,"stable touchdown required");
      fsm.update(.02,.005,false);
      for(int i=0;i<16;++i)fsm.update(.02,0.,true);
      require(fsm.converted & (1<<command_to_foot[command]),"converted foot mapping");
      require(!fsm.request(command),"completed leg rejection");
    }
    require(fsm.converted==15 && fsm.phase==Sequencer::Phase::stand,"all four conversions");
    std::cout<<"PASS: 128 network fixtures; max error "<<network_error<<"; 196 sequential runtime fixtures; max error "<<runtime_error<<"; sequencing and reset contracts\n";
    return 0;
  }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
