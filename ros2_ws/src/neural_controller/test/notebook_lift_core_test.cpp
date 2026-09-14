#include "neural_controller/notebook_alignment_v4/lift_core.hpp"
#include <RTNeural/RTNeural.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
using namespace notebook_alignment_v4;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
std::vector<std::vector<double>> csv(const char* file){std::ifstream f(file);require(bool(f),"fixture file");std::vector<std::vector<double>> out;std::string line;while(std::getline(f,line)){std::stringstream row(line);std::string value;std::vector<double> values;while(std::getline(row,value,','))values.push_back(std::stod(value));out.push_back(values);}return out;}
int main(int argc,char** argv){try{
 require(argc==4,"Expected network JSON, action fixtures, geometry fixtures");
 std::ifstream file(argv[1]);auto net=RTNeural::json_parser::parseJson<float>(file,false);require(net&&net->getInSize()==288&&net->getOutSize()==8,"real network ABI");
 double error=0;auto fixtures=csv(argv[2]);require(fixtures.size()>=128,"At least128 trained inference fixtures");
 for(auto& row:fixtures){require(row.size()==296,"inference fixture width");std::array<float,288> x{};for(int i=0;i<288;++i)x[i]=row[i];net->forward(x.data());for(int i=0;i<8;++i)error=std::max(error,std::abs(double(net->getOutputs()[i])-row[288+i]));}
 std::cout<<"trained_network_fixtures="<<fixtures.size()<<" max_action_error="<<error<<std::endl;require(error<=1e-5,"trained-network inference parity");
 double geometry_error=0;
 for(auto& row:csv(argv[3])){std::array<double,12> q{};std::array<double,3> g{};std::copy_n(row.begin(),12,q.begin());std::copy_n(row.begin()+12,3,g.begin());auto b=WheelGeometry::bottoms(q,g);for(int k=0;k<4;++k)geometry_error=std::max(geometry_error,std::abs(b[k]-row[15+k]));}
 std::cout<<"geometry_max_error="<<geometry_error<<std::endl;require(geometry_error<1e-10,"native model wheel-bottom parity");
 LiftSensors s;for(int i=0;i<8;++i)s.q[PositionTargets::proximal[i]]=PositionTargets::nominal[i];
 for(int k=0;k<4;++k)s.q[PositionTargets::hubs[k]]=40+k*7.;
 LiftCore core;core.reset(s,{40.,47.,54.,61.});auto initial=core.command;
 // State history remains live and targets are absolute, with all eight outputs.
 for(int frame=0;frame<30;++frame){core.prepare(10./520,-1,s);core.targets.set_action({});for(int i=0;i<10;++i)core.execute(1./520,s);}
 require(core.request.phase==0,"idle remains idle without new command");
 core.prepare(10./520,2,s);require(core.request.leg==0&&core.request.phase==1,"externalFR=2 maps canonical0");
 for(int i=0;i<70;++i)core.prepare(10./520,-1,s);
 double h=core.request.height,v=core.request.rate;
 core.prepare(10./520,1,s);require(core.request.phase==3&&core.request.pending==1&&core.request.leg==0,"queueFL and lowerFR first");require(std::abs(core.request.height-h)<.0001,"cancel height continuous");require(std::abs(core.request.rate-v)<.0001,"cancel rate continuous");
 auto elapsed=core.request.elapsed;core.prepare(10./520,0,s);require(core.request.elapsed>elapsed&&core.request.pending==-1,"repeated stand does not reset descent");
 for(int i=0;i<250;++i)core.prepare(10./520,-1,s);require(core.request.phase==0,"supported lowering settles");
 for(int command:{1,2,3,4}){int expected=ObservationHistory::leg_for_external_command(command);core.prepare(10./520,command,s);require(core.request.leg==expected,"all canonical mappings");for(int i=0;i<260;++i)core.prepare(10./520,-1,s);require(core.request.phase==2,"lift reacheshold");core.prepare(10./520,0,s);for(int i=0;i<260;++i)core.prepare(10./520,-1,s);require(core.request.phase==0,"lower returnsstand");}
 core.prepare(10./520,4,s);for(int i=0;i<2550;++i)core.prepare(10./520,-1,s);require(core.request.timed_out&&core.request.phase==3,"48s timeout lowers");for(int i=0;i<300;++i)core.prepare(10./520,-1,s);require(core.request.phase==0&&core.request.timed_out,"timeout doesnotretry");
 for(int i=0;i<500;++i)core.prepare(10./520,-1,s);require(core.request.phase==0,"no unboundedautomaticattempts");core.prepare(10./520,4,s);require(!core.request.timed_out&&core.request.phase==1,"freshrequestretries");
 for(int i=0;i<200;++i){PositionTargets::V8 a{};for(int j=0;j<8;++j)a[j]=(j%2?.2f:-.2f);core.targets.set_action(a);core.execute(1./520,s);for(int k=0;k<4;++k)require(core.command[PositionTargets::hubs[k]]==initial[PositionTargets::hubs[k]],"hubwindingheld");for(int j=0;j<8;++j)require(std::abs(core.targets.velocity[j])<=.100001,"velocitybound");}
 require(std::abs(core.command[0]-1)>0,"actor authority persists");
 bool rejected=false;try{core.targets.set_action({NAN,0,0,0,0,0,0,0});}catch(...){rejected=true;}require(rejected,"rejectnonfiniteaction");
 rejected=false;try{core.execute(.02,s);}catch(...){rejected=true;}require(rejected,"rejectlongcommanddt");
 LiftRequest descent;descent.start(0);descent.advance(4./52,-1,false);descent.lower();for(int i=0;i<750;++i)descent.advance(1./52,-1,false);require(descent.recovery_failed&&descent.phase==3,"no timer-only touchdown");
 std::cout<<"PASS request cancellation/timeouts/held-hub winding/live actor/target bounds; pure software, no hardware evidence\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
