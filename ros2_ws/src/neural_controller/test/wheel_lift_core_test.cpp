#include "neural_controller/wheel_lift/core.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
using namespace wheel_lift;
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(int argc,char** argv){try{
 Sensors s;s.q=Core::nominal;Core c;c.reset(s,{});require(c.stage==Core::idle,"reset");
 require(c.press()&&c.stage==Core::pose,"pose");require(c.press()&&c.stage==Core::lift,"lift");
 require(!c.press()&&c.stage==Core::lift,"closed clearance rejects without queue");
 c.observe(s);auto obs=c.observation;s.q[2]=77;s.qd[2]=3;c.observe(s);require(obs==c.observation,"hub blind");
 s.q[2]=0;s.qd[2]=0;c.ready=true;require(c.press(),"align");c.observe(s);require(obs==c.observation,"align invisible");
 for(int t=0;t<7000&&!c.verified;++t){c.ready=true;c.tick(.004,s);s.q[2]=c.reference[0];s.qd[2]=c.velocity;require(std::abs(c.velocity)<=.15+1e-12,"speed bound");for(int k=1;k<4;++k)require(c.reference[k]==0,"support holds");}
 require(c.verified,"measured settling");require(c.press()&&c.stage==Core::lower,"lower");c.supported=false;require(!c.press(),"reject next while unsupported");c.supported=true;require(c.press()&&c.leg==1,"next leg");
 c.ready=true;c.press();for(int t=0;t<100;++t)c.tick(.004,s);c.ready=false;s.q[5]=.123;c.tick(.004,s);require(c.reference[1]==.123,"gate loss captures once");s.q[5]=.13;c.tick(.004,s);require(c.reference[1]==.123,"does not chase encoder");
 for(int t=0;t<16000;++t)c.tick(.004,s);require(c.stage==Core::fault&&c.actor_command()==1&&!c.press(),"timeout holds lift and rejects descent");
 Sensors bounded;bounded.q=Core::nominal;Core torque;torque.reset(bounded,{});std::array<float,8> extreme;extreme.fill(1);torque.set_action(extreme.data());torque.positions(bounded);for(int i=0;i<12;++i)require(std::abs(torque.estimated_pd[i])<=3.+1e-9,"bounded total PD matches training clamp");require(torque.action==extreme,"saturation does not rewrite previous normalized actor action");
 ActorClock clock;int count=0;for(int t=0;t<5200;++t)count+=clock.tick(1./520);require(count==500,"50 Hz fractional actor schedule");
 s.q=Core::nominal;s.qd.fill(0);for(int k=0;k<4;++k)s.q[3*k+2]=8*M_PI;Core winding;winding.reset(s,{});for(int k=0;k<4;++k)require(std::abs(winding.goal[k]-9*M_PI)<1e-12,"nearest winding half-turn tie");
 require(argc==2,"geometry fixture path");std::ifstream f(argv[1]);std::string line;int rows=0;while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);Sensors x;std::array<double,4> expected;for(auto& v:x.q)in>>v;for(auto& v:x.gravity)in>>v;for(auto& v:expected)in>>v;require(bool(in),"valid fixture");auto actual=WheelGeometry::bottoms(x.q,x.gravity);for(int k=0;k<4;++k)require(std::abs(actual[k]-expected[k])<1e-10,"native MuJoCo geometry parity");++rows;}require(rows==256,"all fixtures");
 std::cout<<"PASS sequence, wheel-blind observation, winding, gate loss, timeout, 50 Hz timing and 256 native geometry fixtures\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
