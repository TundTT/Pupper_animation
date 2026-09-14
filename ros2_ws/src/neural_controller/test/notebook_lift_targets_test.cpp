#include "neural_controller/notebook_alignment_v4/position_targets.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
using notebook_alignment_v4::PositionTargets;
int main(int argc,char**argv){if(argc!=2)return 2;PositionTargets controller;std::ifstream f(argv[1]);std::string line;double error=0;int count=0;while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);PositionTargets::V8 action,target,velocity;std::array<float,4> hubs;float dt;for(auto&x:action)if(!(row>>x))return 3;if(!(row>>dt))return 3;for(auto&x:hubs)if(!(row>>x))return 3;for(auto&x:target)if(!(row>>x))return 3;for(auto&x:velocity)if(!(row>>x))return 3;controller.set_action(action);auto command=controller.step(dt,hubs);for(int i=0;i<8;++i){error=std::max(error,double(std::abs(target[i]-controller.applied[i])));error=std::max(error,double(std::abs(velocity[i]-controller.velocity[i])));}for(int i=0;i<4;++i)if(command[PositionTargets::hubs[i]]!=hubs[i])return 4;++count;}std::cout<<"steps="<<count<<" max_error="<<error<<'\n';return count==20000&&error<=1e-5?0:5;}
