#include "controller.hpp"
extern "C" {
void* kf_create(const double* values,const double* poses){
  try {auto c=new keyframe_align::Controller;
    if(values)std::copy_n(values,15,c->config.values.begin());
    if(poses)for(int k=0;k<4;++k)std::copy_n(poses+8*k,8,c->config.poses[k].begin());
    try{c->config.validate();}catch(...){delete c;return nullptr;}return c;
  }catch(...){return nullptr;}
}
void kf_destroy(void* ptr){delete static_cast<keyframe_align::Controller*>(ptr);}
int kf_reset(void* ptr,const double* q,const double* home){
  try{keyframe_align::V12 a{};std::array<double,4>b{};std::copy_n(q,12,a.begin());std::copy_n(home,4,b.begin());
    static_cast<keyframe_align::Controller*>(ptr)->reset(a,b);return 0;}catch(...){return -1;}
}
void kf_step(void* ptr,double dt,int command,const double* q,const double* qd,const double* angular,const double* gravity,int stop,double* out){
  keyframe_align::V12 a{},b{};keyframe_align::V3 w{},g{};
  std::copy_n(q,12,a.begin());std::copy_n(qd,12,b.begin());std::copy_n(angular,3,w.begin());std::copy_n(gravity,3,g.begin());
  auto o=static_cast<keyframe_align::Controller*>(ptr)->step(dt,command,a,b,w,g,stop);
  std::copy(o.position.begin(),o.position.end(),out);std::copy(o.wheel.begin(),o.wheel.end(),out+8);
  out[12]=o.phase;out[13]=o.active;out[14]=o.completed;out[15]=o.blocked;out[16]=o.timeout;
  std::copy(o.margins.begin(),o.margins.end(),out+17);out[20]=o.error;out[21]=o.up_seconds;
  out[22]=o.integral;out[23]=o.authority;
}
}
