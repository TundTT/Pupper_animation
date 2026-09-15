#include "neural_controller/notebook_alignment_v4/lift_align_core.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
using namespace notebook_alignment_v4;
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
std::vector<std::vector<double>> csv(const char* path){std::ifstream f(path);require(bool(f),"fixture exists");std::string line;std::vector<std::vector<double>> rows;while(std::getline(f,line)){std::stringstream s(line);std::string v;std::vector<double> r;while(std::getline(s,v,','))r.push_back(std::stod(v));rows.push_back(r);}return rows;}
int main(int argc,char** argv){try{
 require(argc==2,"Expected native geometry fixtures");auto rows=csv(argv[1]);require(rows.size()==1028,"native fixtures and four reachable kinematic poses");double max_error=0;
 for(auto& r:rows){require(r.size()==19,"geometry width");std::array<double,12> q{};std::array<double,3> g{};std::copy_n(r.begin(),12,q.begin());std::copy_n(r.begin()+12,3,g.begin());auto m=AlignGeometry::margins(q,g,int(r[15]));for(int i=0;i<3;++i)max_error=std::max(max_error,std::abs(m[i]-r[16+i]));}
 require(max_error<1e-10,"native full geometry parity");
 const double pi=std::acos(-1.),dt=10./520,sub=1./520;
 for(double home:{-100.,-pi,-.2,0.,pi,100.}){
  require(std::abs(LiftAlignCore::goal_near(home,home)-home-pi)<1e-12,"exact startup plus positive180 tie");
  for(double current:{home+pi,home-pi,home+14*pi+.1,home-18*pi-.1}){
   double goal=LiftAlignCore::goal_near(home,current);require(std::abs(std::remainder(goal-home-pi,2*pi))<1e-12,"goal equivalent to startup+180");require(std::abs(goal-current)<=pi+1e-12,"nearest winding, never long spin");
  }
 }
 LiftSensors stand;for(int a=0;a<8;++a)stand.q[PositionTargets::proximal[a]]=PositionTargets::nominal[a];
 LiftCore baseline;LiftAlignCore disabled;baseline.reset(stand,{});disabled.reset(stand,{});
 for(int frame=0;frame<300;++frame){int event=frame==30?2:frame==200?0:-1;baseline.prepare(dt,event,stand);disabled.prepare(dt,event,stand);PositionTargets::V8 a{};a.fill(.01);baseline.targets.set_action(a);disabled.targets.set_action(a);for(int tick=0;tick<10;++tick){baseline.execute(sub,stand);disabled.execute(sub,stand);}require(baseline.command==disabled.command&&baseline.history.values==disabled.history.values,"disabled alignment preserves lift-only ABI/execution exactly");}
 LiftAlignCore c;c.alignment_enabled=true;std::array<double,4> homes{0,.3,-.6,1.2};for(int k=0;k<4;++k)stand.q[3*k+2]=homes[k];c.reset(stand,homes);
 for(int i=0;i<30;++i)c.prepare(dt,-1,stand);c.prepare(dt,5,stand);require(c.request.phase==0&&!c.rotation_requested,"rotate button cannot rotate in stand");
 // Ideal encoder-following unit harness: exercises servo/state, NOT balance.
 for(int k=0;k<4;++k){
  const int command= k==0?2:k==1?1:k+1;c.prepare(dt,command,stand);require(c.request.leg==k&&c.request.phase==1,"selected leg mapping");
  c.prepare(dt,5,stand);require(c.rotation_requested&&!c.rotation_enabled,"early rotate queues until clearance and hold");
  LiftSensors air=stand;for(int a=0;a<8;++a)air.q[PositionTargets::proximal[a]]=rows[1024+k][PositionTargets::proximal[a]];
  PositionTargets::V8 pose{};for(int a=0;a<8;++a)pose[a]=air.q[PositionTargets::proximal[a]];c.targets.reset(pose);
  for(int i=0;i<210;++i)c.prepare(dt,-1,air);
  require(c.request.phase==2,"lift requested height reached");
  auto before=c.hold;bool rotated=false;double maxspeed=0,maxaccel=0,lastv=0;
  for(int t=0;t<1500&&!c.verified;++t){
   c.prepare(dt,-1,air);double v=c.hub_velocity[k];maxspeed=std::max(maxspeed,std::abs(v));if(!c.verified)maxaccel=std::max(maxaccel,std::abs(v-lastv)/dt);lastv=v;
   require(c.history.values[51]==float(c.rotation_enabled),"upcoming rotation permission in actor frame");require(std::abs(c.history.values[52+k]-v)<1e-7,"upcoming speed in actor frame");
   for(int j=0;j<10;++j){c.execute(sub,air);for(int h=0;h<4;++h){double next=c.command[3*h+2];air.qd[3*h+2]=(next-air.q[3*h+2])/sub;air.q[3*h+2]=next;if(h!=k)require(next==before[h],"supporting hubs keep exact holds");}}
   rotated|=std::abs(air.q[3*k+2]-before[k])>.1;
  }
  require(rotated&&c.verified,"all four half-turns settle");require(maxspeed<=.5000001&&maxaccel<=1.2000001,"normal reference speed/acceleration limits");require(std::abs(c.hold[k]-homes[k]-pi)<1e-12,"holds exact startup+pi after verification");require(c.completed==(1<<k)-1,"no completion before landing");
  auto goal=c.goal;c.prepare(dt,5,air);require(!c.rotation_requested&&c.goal==goal,"repeat rotate cannot add another halfturn");
  c.prepare(dt,0,air);require(c.stopping&&c.request.phase==2,"lower waits actual stopped hub");
  air.qd.fill(0);for(int i=0;i<12;++i)c.prepare(dt,-1,air);require(c.request.phase==3,"settled hub permits descent");
  for(int h=0;h<4;++h)stand.q[3*h+2]=c.hold[h];c.targets.reset(PositionTargets::nominal);
  for(int i=0;i<250;++i)c.prepare(dt,-1,stand);require(c.request.phase==0,"supported lowered stand");require(c.completed==(1<<(k+1))-1,"only actual aligned lowered leg completes");require(c.goal==goal,"goal invariant through lowering");
  c.prepare(dt,command,stand);require(c.request.phase==0,"completed leg request cannot spin again");
 }
 require(c.completed==15,"all four completed masks retained");stand.q[2]+=.04;c.prepare(dt,-1,stand);require(!(c.completed&1)&&(c.failed&1),"drift invalidates completion without moving goal");
 // Gate loss is checked at command substeps and captures exactly once.
 c.reset(stand,homes);c.request.start(0);c.request.phase=2;c.request.height=.008;
 LiftSensors air=stand;for(int a=0;a<8;++a)air.q[PositionTargets::proximal[a]]=rows[1024][PositionTargets::proximal[a]];
 PositionTargets::V8 pose{};for(int a=0;a<8;++a)pose[a]=air.q[PositionTargets::proximal[a]];c.targets.reset(pose);
 c.prepare(dt,5,air);for(int t=0;t<20;++t)c.prepare(dt,-1,air);require(c.rotation_enabled,"qualified gate opens");
 air.angular={1,0,0};c.execute(sub,air);require(!c.rotation_enabled&&c.hub_velocity[0]==0&&c.hold[0]==air.q[2],"substep loss captures hold and stops advancement");double hold=c.hold[0];air.q[2]+=.001;c.execute(sub,air);require(c.hold[0]==hold,"closed gate never chases encoder");
 air.angular={0,0,0};for(int i=0;i<5;++i)c.prepare(dt,-1,air);require(!c.rotation_enabled,"loss resets full qualification duration");
 for(int i=0;i<10;++i)c.prepare(dt,-1,air);require(c.rotation_enabled,"new qualification resumes toward same goal");
 double h=c.request.height;c.prepare(dt,1,air);require(c.stopping&&c.queued==1&&c.request.leg==0&&c.request.height==h,"NEXT brakes before lowering current leg");
 air.qd[2]=.2;for(int i=0;i<10;++i)c.prepare(dt,0,air);require(c.request.phase==2&&c.queued==0,"repeated stand cancels queued leg without descent while spinning");
 air.qd[2]=0;for(int i=0;i<12;++i)c.prepare(dt,-1,air);require(c.request.phase==3&&!c.verified,"unverified cancellation lowers without false completion");
 for(int k=0;k<4;++k)stand.q[3*k+2]=c.hold[k];c.targets.reset(PositionTargets::nominal);for(int i=0;i<250;++i)c.prepare(dt,-1,stand);require(c.completed==0&&c.failed==1,"interrupted wheel remains incomplete");
 c.prepare(dt,2,stand);c.request.attempt=47.99;c.prepare(dt,5,stand);require(c.request.timed_out&&c.stopping,"48s timeout brakes before lowering");for(int i=0;i<270;++i)c.prepare(dt,-1,stand);require(c.request.phase==0&&c.request.timed_out,"timeout recovery finishes without retry");for(int i=0;i<100;++i)c.prepare(dt,-1,stand);require(c.request.phase==0,"no automatic timed-out attempts");
 for(int k=0;k<4;++k)stand.q[3*k+2]=homes[k]+pi+8*pi;c.reset(stand,homes);for(int k=0;k<4;++k)require(std::abs(c.goal[k]-stand.q[3*k+2])<1e-12,"reentry aligned after revolutions does not rotate another180");
 c.request.start(0);c.request.phase=2;c.request.height=.008;c.request.attempt=47.99;stand.qd[2]=.2;c.prepare(dt,5,stand);bool rejected=false;try{for(int i=0;i<120;++i)c.prepare(dt,0,stand);}catch(const std::runtime_error&){rejected=true;}require(rejected&&c.request.phase==2,"timeout does not descend while hub fails to stop");
 std::cout<<"PASS 1028 native geometry fixtures max_error="<<max_error<<"; all4 startup+pi goals, winding, normal servo limits, completion/holds, substep gate loss, cancellation, timeout/reentry. Ideal encoder unit harness, not learned balance.\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
