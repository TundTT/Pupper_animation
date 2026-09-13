#include "neural_controller/triangle_roll.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv) {
  try {
    check(argc==2,"fixture path required");std::ifstream f(argv[1]);check(bool(f),"fixture missing");
    triangle_roll::Support support;std::string line;int rows=0;double peak=0;
    while(std::getline(f,line)) {
      std::stringstream s(line);std::string token;std::vector<double> values;
      while(std::getline(s,token,','))values.push_back(std::stod(token));check(values.size()==55,"fixture shape");
      triangle_roll::Pose q,qd,previous;
      for(int i=0;i<12;++i){q[i]=values[3+i];qd[i]=values[15+i];previous[i]=values[27+i];}
      support.update(q,qd,previous,{values[0],values[1],values[2]},10./520);
      for(int i=0;i<12;++i)peak=std::max(peak,std::abs(support.offset[i]-values[39+i]));
      for(int i=0;i<4;++i)peak=std::max(peak,std::abs(support.loads[i]-values[51+i]));
      ++rows;
    }
    std::cerr<<"parity max "<<peak<<" rows "<<rows<<"\n";check(rows==520 && peak<1e-8,"C++ support differs from pinned Python/MuJoCo reference");
    using namespace inverted_triangle;triangle_roll::Controller c;Pose q=triangle_roll::home,qd{},map{};
    map[2]=.7;map[5]=6.283185307179586;for(int i=0;i<12;++i)q[i]+=map[i];
    c.reset(q,map);for(int k=0;k<1041;++k)c.step(1./520,q,qd,{0,0,-1},1,k==0);
    check(c.state==READY,"early command must not start roll");
    c.step(1./520,q,qd,{0,0,-1},1,true);
    for(int k=0;k<23921 && c.state==RUNNING;++k) {
      for(int i=0;i<12;++i){double next=c.command[i]+map[i];qd[i]=(next-q[i])*520;q[i]=next;}
      c.step(1./520,q,qd,{0,0,-1});check(c.state!=FAULT,"ideal tracking roll unexpectedly faults");
    }
    check(c.state==DONE && c.completed==1,"roll must finish with no walking stage");
    Pose frozen=c.command;qd.fill(0);for(int k=0;k<20;++k)c.step(1./520,q,qd,{0,0,-1},1,true);
    check(c.command==frozen && c.state==DONE,"done must hold, not repeat");
    c.step(1./520,q,qd,{0,0,-1},-1,true);check(c.state==FAULT && c.gain==0,"stop must release torque");
    c.reset(q=triangle_roll::home,Pose{});c.step(.02,q,qd,{0,0,-1});check(c.fault==TIMING,"clock guard");
    c.reset(q,Pose{});c.step(1./520,q,qd,{0,.2,-std::sqrt(.96)});check(c.fault==TILT,"tilt guard");
    c.reset(q,Pose{});qd[2]=2.1;c.step(1./520,q,qd,{0,0,-1});check(c.fault==SPEED,"speed guard");qd.fill(0);
    c.reset(q,Pose{});c.step(1./520,q,qd,{0,0,-1},2,true);check(c.fault==COMMAND,"reject old sequential commands");
    c.reset(q,Pose{});q[0]+=.26;c.step(1./520,q,qd,{0,0,-1});check(c.fault==TRACKING,"tracking guard");
    std::cout<<"PASS: "<<rows<<" Python/MuJoCo fixtures, max error "<<peak<<", mapped continuous roll, hold and guards\n";
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
