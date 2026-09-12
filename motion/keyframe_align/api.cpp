#include "controller.hpp"
extern "C" {
int kf_abi_version(){return 3;}
void kf_margins(const double* q,const double* gravity,int leg,double* out){
  keyframe_align::V12 a{};keyframe_align::V3 g{};
  std::copy_n(q,12,a.begin());std::copy_n(gravity,3,g.begin());
  auto m=keyframe_align::Geometry::margins(a,g,leg);std::copy(m.begin(),m.end(),out);
}
void* kf_create(const double* values,const double* poses,const double* settings){
  try {auto c=new keyframe_align::Controller;
    if(values)std::copy_n(values,15,c->config.values.begin());
    if(poses)for(int k=0;k<4;++k)std::copy_n(poses+8*k,8,c->config.poses[k].begin());
    if(settings){
    auto& f=c->config;
    f.rotation_floor_clearance_m=settings[0];f.wheel_position_kp=settings[1];f.wheel_position_kd=settings[2];
    f.alignment_angle_tolerance_rad=settings[3];f.landing_angle_tolerance_rad=settings[4];
    f.alignment_speed_tolerance_rad_s=settings[5];f.alignment_settle_seconds=settings[6];f.hold_error_limit_rad=settings[7];
    f.wheel_integral_ki=settings[8];f.wheel_integral_limit_nm=settings[9];f.wheel_integral_window_rad=settings[10];
    }
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
  out[22]=o.integral;out[23]=o.authority;out[24]=o.failed;
  std::copy(o.wheel_position.begin(),o.wheel_position.end(),out+25);
  std::copy(o.wheel_effort.begin(),o.wheel_effort.end(),out+29);
}
}
