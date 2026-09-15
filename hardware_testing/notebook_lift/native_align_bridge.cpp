// Native simulation adapter for the actual runtime core + unchanged trained actor.
#include "neural_controller/notebook_alignment_v4/lift_align_core.hpp"
#include <RTNeural/RTNeural.h>
#include <fstream>
#include <cstring>
using namespace notebook_alignment_v4;
struct Native {LiftAlignCore core;std::unique_ptr<RTNeural::Model<float>> net;int ticks=0;double elapsed=0;std::string error;};
LiftSensors sensors(const double* q,const double* qd,const double* g,const double* w){LiftSensors s;std::copy_n(q,12,s.q.begin());std::copy_n(qd,12,s.qd.begin());std::copy_n(g,3,s.gravity.begin());std::copy_n(w,3,s.angular.begin());return s;}
extern "C" {
void* align_create(const char* path){try{auto n=std::make_unique<Native>();std::ifstream f(path);n->net=RTNeural::json_parser::parseJson<float>(f,false);if(!n->net||n->net->getInSize()!=288||n->net->getOutSize()!=8)return nullptr;n->core.alignment_enabled=true;return n.release();}catch(...){return nullptr;}}
void align_destroy(void* p){delete static_cast<Native*>(p);}
const char* align_error(void* p){return static_cast<Native*>(p)->error.c_str();}
int align_reset(void* p,const double* q,const double* qd,const double* g,const double* w,const double* home){auto& n=*static_cast<Native*>(p);try{std::array<double,4> h{};std::copy_n(home,4,h.begin());n.core.reset(sensors(q,qd,g,w),h);n.ticks=0;n.elapsed=0;n.net->reset();n.error.clear();return 0;}catch(const std::exception& e){n.error=e.what();return 1;}}
int align_tick(void* p,const double* q,const double* qd,const double* g,const double* w,int event,double* output){auto& n=*static_cast<Native*>(p);try{auto s=sensors(q,qd,g,w);n.elapsed+=1./520;
 if(n.ticks++%10==0){n.core.prepare(n.elapsed,event,s);n.elapsed=0;n.net->forward(n.core.history.values.data());PositionTargets::V8 action{};std::copy_n(n.net->getOutputs(),8,action.begin());n.core.targets.set_action(action);}
 n.core.execute(1./520,s);std::copy(n.core.command.begin(),n.core.command.end(),output);return 0;}catch(const std::exception& e){n.error=e.what();return 1;}}
void align_status(void* p,double* v){auto& c=static_cast<Native*>(p)->core;v[0]=c.request.phase;v[1]=c.request.leg;v[2]=c.rotation_requested;v[3]=c.rotation_enabled;v[4]=c.verified;v[5]=c.completed;v[6]=c.request.timed_out;v[7]=c.request.height;std::copy(c.margins.begin(),c.margins.end(),v+8);std::copy(c.goal.begin(),c.goal.end(),v+11);std::copy(c.hold.begin(),c.hold.end(),v+15);std::copy(c.hub_velocity.begin(),c.hub_velocity.end(),v+19);std::copy(c.targets.requested.begin(),c.targets.requested.end(),v+23);std::copy(c.targets.applied.begin(),c.targets.applied.end(),v+31);std::copy(c.estimated_pd.begin(),c.estimated_pd.end(),v+39);}
}
