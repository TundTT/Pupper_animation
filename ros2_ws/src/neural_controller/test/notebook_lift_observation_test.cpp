#include "neural_controller/notebook_alignment_v4/observation.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc,char**argv){
 if(argc!=2)return 2;using namespace notebook_alignment_v4;ObservationHistory history;std::ifstream in(argv[1]);std::string line;float error=0;int count=0;
 while(std::getline(in,line)){
  std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);ObservationInput s;int reset,enabled;row>>reset;
  for(auto&v:s.angular)row>>v;for(auto&v:s.gravity)row>>v;row>>s.leg>>s.phase;
  for(auto&v:s.position)row>>v;for(auto&v:s.velocity)row>>v;for(auto&v:s.home)row>>v;for(auto&v:s.reference)row>>v;for(auto&v:s.hub_velocity)row>>v;for(auto&v:s.applied)row>>v;for(auto&v:s.target_velocity)row>>v;row>>s.desired_clearance>>enabled;s.rotation_enabled=enabled;
  if(reset)history.reset(s);else history.update(s);
  for(float v:history.values){float expected;row>>expected;if(!row)return 3;error=std::max(error,std::abs(v-expected));}++count;
 }
 if(ObservationHistory::leg_for_external_command(1)!=1||ObservationHistory::leg_for_external_command(2)!=0||ObservationHistory::leg_for_external_command(0)!=-1)return 4;
 std::cout<<"frames="<<count<<" maximum_observation_error="<<error<<"\n";return count>=128&&error<1e-5f?0:1;
}
