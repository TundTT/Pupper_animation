#pragma once
#include "neural_controller/wheel_align_hybrid.hpp"
#include "neural_controller/wheel_align_geometry_data.hpp"
#include "neural_controller/wheel_align_reference_data.hpp"
#include "neural_controller/policy_home.hpp"

namespace neural_controller {
// Version 5 supplies coordinated reference poses and supported descent.
struct WheelAlignMotion {
  static constexpr int version = 5;
  static constexpr const char* contract_id = "quadmorph-align-motion-v5";
  static constexpr int observation_size = 83;
  static constexpr double control_dt = 10.0/520.0;
  static constexpr std::array<double,8> neutral=policy_home::upper;
  static constexpr std::array<double,8> low{-1.12,-.32,-2.41,-3.04,-1.12,-.32,-2.41,-3.04};
  static constexpr std::array<double,8> high{2.41,3.04,1.12,.32,2.41,3.04,1.12,.32};
  std::array<double,8> reference=neutral, start=neutral, applied=neutral, velocity{}, desired=neutral;
  int previous_phase=WheelAlignHybrid::IDLE;
  double elapsed=0, progress=0, residual_gain=1;
  static double smooth(double u) { u=std::clamp(u,0.,1.); return u*u*u*(10+u*(-15+6*u)); }
  void reset(const std::array<double,12>& q) {
    *this=WheelAlignMotion{};
    for(int a=0;a<8;++a) reference[a]=start[a]=applied[a]=desired[a]=q[WheelAlignHybrid::position_rows[a]];
  }
  void prepare(const WheelAlignHybrid& h, double dt) {
    const bool previous_up=previous_phase>=WheelAlignHybrid::LIFT && previous_phase<=WheelAlignHybrid::VERIFY;
    if (h.phase!=previous_phase && !(h.up() && previous_up)) {
      previous_phase=h.phase; elapsed=0; start=applied;
      if(h.phase==WheelAlignHybrid::LIFT) residual_gain=1;
    }
    previous_phase=h.phase;
    elapsed+=std::clamp(dt,0.,.04);
    using namespace align_reference;
    const double duration=h.phase==WheelAlignHybrid::LOWER ? lower_seconds : lift_seconds;
    progress=std::min(elapsed/duration,1.);
    auto apex=poses[h.leg()], shift=apex, landing=start;
    const int hip=2*h.leg()+1; const double sign=h.leg()%2==0 ? 1. : -1.;
    shift[hip]=land_hip*sign;
    landing[hip]=sign*std::min(std::max(sign*start[hip],0.),land_hip);
    for(int a=0;a<8;++a) {
      if(h.up()) reference[a]=elapsed<shift_seconds ?
        start[a]+smooth(elapsed/shift_seconds)*(shift[a]-start[a]) :
        shift[a]+smooth((elapsed-shift_seconds)/rise_seconds)*(apex[a]-shift[a]);
      else if(h.phase==WheelAlignHybrid::LOWER) reference[a]=elapsed<land_seconds ?
        start[a]+smooth(elapsed/land_seconds)*(landing[a]-start[a]) :
        landing[a]+smooth((elapsed-land_seconds)/recenter_seconds)*(neutral[a]-landing[a]);
      else reference[a]=neutral[a];
    }
  }
  std::array<double,8> targets(const WheelAlignHybrid& h, const std::array<double,8>& action,
      bool comfortable=true, double dt=control_dt) {
    using namespace align_reference;
    if(h.up() && progress>=1 && (residual_gain<1 || !comfortable)) {
      const double previous=residual_gain;
      residual_gain=std::max(previous-std::clamp(dt,0.,.04)/recovery_seconds,0.);
      for(int a=0;a<8;++a) desired[a]=reference[a]+(desired[a]-reference[a])*residual_gain/std::max(previous,1e-9);
      return desired;
    }
    if(h.phase==WheelAlignHybrid::VERIFY) return desired;
    for(int a=0;a<8;++a) {
      const bool active=a/2==h.leg();
      double scale=(active ? active_residual[a%2] : support_residual[a%2]);
      scale*=h.up() ? smooth(std::min(elapsed/shift_seconds,1.)) : 0.;
      const double sign=h.leg()%2==0 ? 1. : -1.;
      const double residual=(active && a%2==1 && h.up()) ? sign*std::max(sign*action[a],0.) : action[a];
      desired[a]=std::clamp(reference[a]+scale*residual,low[a],high[a]);
    }
    return desired;
  }

  void integrate(const WheelAlignHybrid& h, double dt) {
    dt=std::clamp(dt,0.,.01);
    for(int a=0;a<8;++a) {
      const bool active=(a/2==h.leg())&&(h.up()||h.phase==WheelAlignHybrid::LOWER);
      const double vmax=active ? align_reference::active_speed[a%2] : align_reference::support_speed[a%2];
      const double accel=active ? align_reference::active_accel[a%2] : align_reference::support_accel[a%2];
      // Decelerate to a newly smaller speed limit without resetting velocity.
      const double dv=std::clamp(36*(desired[a]-applied[a])-12*velocity[a],-accel,accel)*dt;
      const double bound=std::max(vmax,std::abs(velocity[a])-accel*dt);
      velocity[a]=std::clamp(velocity[a]+dv,-bound,bound);
      applied[a]+=velocity[a]*dt;
      if(applied[a]<low[a] || applied[a]>high[a]) {
        applied[a]=std::clamp(applied[a],low[a],high[a]); velocity[a]=0;
      }
    }
  }
  bool lower_finished(const WheelAlignHybrid& h) const {
    return progress>=1 && std::abs(velocity[2*h.leg()+1])<.03 &&
        std::abs(applied[2*h.leg()+1])<.03;
  }

  using Vec=std::array<double,3>;
  using Mat=std::array<double,9>;
  static Mat mul(const Mat& a,const Mat& b) {
    Mat r{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) r[3*i+j]+=a[3*i+k]*b[3*k+j]; return r;
  }
  static Vec mv(const Mat& a,const Vec& b) {
    Vec r{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) r[i]+=a[3*i+j]*b[j]; return r;
  }
  static Mat rz(double q) {return {std::cos(q),-std::sin(q),0,std::sin(q),std::cos(q),0,0,0,1};}
  static double norm(const Vec& v){return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
  static std::array<double,3> margins(const std::array<double,12>& q,const Vec& gravity,int leg) {
    using namespace align_geometry;
    constexpr double radius=.0505, half=.01675;
    const double bound=std::hypot(radius,half);
    std::array<Vec,4> centers{}; std::array<double,4> bottom{};
    double wheel_gap=1e3,box_gap=1e3,support_bottom=-1e3;
    for(int k=0;k<4;++k) {
      Mat a{},b{},c{};
      std::copy_n(r1.begin()+9*k,9,a.begin()); std::copy_n(r2.begin()+9*k,9,b.begin()); std::copy_n(r3.begin()+9*k,9,c.begin());
      const auto r=mul(mul(mul(a,rz(q[3*k])),b),rz(q[3*k+1]));
      auto p=mv(r,{p3[3*k],p3[3*k+1],p3[3*k+2]}); const auto orientation=mul(r,c);
      double z=0,dot=0;
      for(int i=0;i<3;++i) {centers[k][i]=p[i]+p1[3*k+i]+wheel_center_z[k]*orientation[3*i+2]; z-=centers[k][i]*gravity[i]; dot-=orientation[3*i+2]*gravity[i];}
      bottom[k]=z-radius*std::sqrt(std::max(1-dot*dot,0.))-half*std::abs(dot);
      if(k!=leg) support_bottom=std::max(support_bottom,bottom[k]);
      Vec local{};
      for(int i=0;i<3;++i) {
        for(int j=0;j<3;++j) local[i]+=(centers[k][j]-box_pos[j])*box_rot[3*j+i];
        local[i]=std::max(std::abs(local[i])-box_size[i],0.);
      }
      box_gap=std::min(box_gap,norm(local)-bound);
    }
    for(int a=0;a<4;++a) for(int b=a+1;b<4;++b) {
      Vec d{};for(int j=0;j<3;++j)d[j]=centers[a][j]-centers[b][j]; wheel_gap=std::min(wheel_gap,norm(d)-2*bound);
    }
    return {bottom[leg]-support_bottom,wheel_gap,box_gap};
  }
  bool ready(const WheelAlignHybrid& h,const std::array<double,12>& q,const Vec& angular,const Vec& gravity) const {
    const auto m=margins(q,gravity,h.leg());
    return progress>=1 && m[0]>.010 && m[1]>.010 && m[2]>.005 && -gravity[2]>std::cos(.12) && norm(angular)<.3;
  }
  bool comfortable(const WheelAlignHybrid& h,const std::array<double,12>& q,const Vec& angular,const Vec& gravity) const {
    const auto m=margins(q,gravity,h.leg());
    return m[0]>align_reference::recovery_floor && m[1]>align_reference::recovery_wheel &&
      m[2]>align_reference::recovery_body && ready(h,q,angular,gravity);
  }
};
}
