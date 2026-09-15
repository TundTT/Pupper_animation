// Native simulation bridge runs the same Core and RTNeural graph as the ROS plugin.
#include "neural_controller/wheel_lift/core.hpp"
#include <RTNeural/RTNeural.h>
#include <fstream>
#include <memory>
#include <string>
struct Bridge {wheel_lift::Core core;wheel_lift::ActorClock clock;std::unique_ptr<RTNeural::Model<float>> model;std::string error;};
extern "C" {
void* wheel_create(const char* path){try{auto b=std::make_unique<Bridge>();std::ifstream f(path);b->model=RTNeural::json_parser::parseJson<float>(f,false);if(!b->model||b->model->getInSize()!=27||b->model->getOutSize()!=8)return nullptr;return b.release();}catch(...){return nullptr;}}
void wheel_destroy(void* p){delete static_cast<Bridge*>(p);}
const char* wheel_error(void* p){return static_cast<Bridge*>(p)->error.c_str();}
int wheel_step(void* p,double dt,const double* values,int event,double* out){auto& b=*static_cast<Bridge*>(p);try{
 wheel_lift::Sensors s;std::copy_n(values,12,s.q.begin());std::copy_n(values+12,12,s.qd.begin());std::copy_n(values+24,3,s.angular.begin());std::copy_n(values+27,3,s.gravity.begin());
 if(event==2){b.core.reset(s,{});b.core.press();b.clock={};}
 b.core.measure(s);b.core.tick(dt,s);if(event==1){b.core.press();b.core.measure(s);}
 if(b.clock.tick(dt)){b.core.observe(s);b.model->forward(b.core.observation.data());b.core.set_action(b.model->getOutputs());}
 b.core.positions(s);std::copy(b.core.command.begin(),b.core.command.end(),out);
 out[12]=b.core.stage;out[13]=b.core.leg;out[14]=b.core.supported;out[15]=b.core.ready;out[16]=b.core.verified;
 std::copy(b.core.margins.begin(),b.core.margins.end(),out+17);return 0;
 }catch(const std::exception& e){b.error=e.what();return 1;}}
void wheel_actor(void* p,const float* in,float* out){auto& b=*static_cast<Bridge*>(p);b.model->forward(in);std::copy_n(b.model->getOutputs(),8,out);}
}
