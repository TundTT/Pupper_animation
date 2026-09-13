#pragma once
#include "neural_controller/inverted_triangle.hpp"
#include "neural_controller/triangle_roll_geometry.hpp"
#include "neural_controller/policy_home.hpp"

namespace triangle_roll {
using inverted_triangle::Pose;
using inverted_triangle::smooth;
inline constexpr Pose goal=neural_controller::policy_home::with_hubs({-1,1,-1,1});
inline constexpr Pose home{1,-.29,2.141592653589793,-1,.29,-2.141592653589793,1,-.29,2.141592653589793,-1,.29,-2.141592653589793};
inline constexpr Pose base_kp{5,5,4,5,5,4,5,5,4,5,5,4};
inline constexpr Pose base_kd{.25,.25,.15,.25,.25,.15,.25,.25,.15,.25,.25,.15};

// Exact nominal gravity/Jacobian calculation for the 12 hinges. This is a
// model-based PD load estimate, not contact sensing. Input gravity is body-frame
// R^T*[0,0,-1], already normalized by the shared IMU reader. Yaw cancels out.
struct Support {
  Pose reference=goal,integral{},offset{};
  Pose target=goal;
  void set_goal(const Pose& q) { target=q;reference=q;integral.fill(0);offset.fill(0);loads.fill(0);elapsed=0; }
  std::array<double,4> loads{};
  double elapsed=0;
  void update(const Pose& q,const Pose& qd,const Pose& previous,const V3& gravity,double dt) {
    elapsed+=dt;
    std::array<V3,4> zs;
    const V3 up=-gravity;
    for(int leg=0;leg<4;++leg) {
      std::array<V3,3> origins,axes,coms;
      Eigen::Matrix3d R=Eigen::Matrix3d::Identity();V3 p=V3::Zero();
      for(int j=0;j<3;++j) {
        const auto& link=links[3*leg+j];p+=R*link.pos;R=R*link.rot.toRotationMatrix();
        origins[j]=p;axes[j]=R*link.axis;
        R=R*Eigen::AngleAxisd(q[3*leg+j]-link.ref,link.axis).toRotationMatrix();
        coms[j]=p+R*link.com;
      }
      V3 point=p+R*tips[leg][0];double minimum=up.dot(point);
      for(const auto& local:tips[leg]) {const V3 candidate=p+R*local;double h=up.dot(candidate);if(h<minimum){minimum=h;point=candidate;}}
      V3 external;
      for(int j=0;j<3;++j) {
        const int n=3*leg+j;zs[leg][j]=up.dot(axes[j].cross(point-origins[j]));
        double bias=0;
        for(int b=j;b<3;++b) bias+=9.81*links[3*leg+b].mass*up.dot(axes[j].cross(coms[b]-origins[j]));
        external[j]=bias-(base_kp[n]*(previous[n]-q[n])-base_kd[n]*qd[n]);
      }
      double projected=zs[leg].dot(external)/(zs[leg].squaredNorm()+1e-8);
      double blend=std::clamp((elapsed-2.)/4.,0.,1.);
      double estimate=(1-blend)*projected+blend*external.norm()/std::max(zs[leg].norm(),.01);
      loads[leg]+=(1-std::exp(-dt/.4))*(estimate-loads[leg]);
    }
    double desired=0;for(double f:loads)desired+=std::clamp(f,0.,30.)/4.;
    for(int leg=0;leg<4;++leg) {
      const double error=std::clamp(desired-loads[leg],-6.,6.);
      for(int j=0;j<3;++j)reference[3*leg+j]-=.003*error*zs[leg][j]/std::max(zs[leg].norm(),.01)*dt;
    }
    for(int i=0;i<12;++i) {
      reference[i]=std::clamp(reference[i],target[i]-.075,target[i]+.075);
      integral[i]=std::clamp(integral[i]+.25*(reference[i]-q[i])*dt,-.15,.15);
      offset[i]=std::clamp(reference[i]+integral[i]-target[i],-.2,.2);
    }
  }
};

// Hardware-only envelope around the pinned 46-second roll+settle simulation.
// READY waits for command 1. DONE holds the last command and does not walk.
struct Controller {
  Pose initial=home,kp=base_kp,kd=base_kd,offset{},command=home,entry{},velocity{};
  Support support;
  bool stand_only=false;  // Diagnostic roll+hold, with no ground-load adaptation.
  inverted_triangle::State state=inverted_triangle::FAULT;
  inverted_triangle::Fault fault=inverted_triangle::ACTIVATION;
  int segment=0,completed=0,run_steps=0;
  double elapsed=0,ramp=0,gain=0,gate_wait=0,max_error=0,max_torque=0;
  void validate() const {
    if(initial!=home || kp!=base_kp)throw std::runtime_error("Unexpected roll pose/gains");
  }
  void stop(inverted_triangle::Fault why) {if(state!=inverted_triangle::FAULT)fault=why;state=inverted_triangle::FAULT;gain=0;}
  void reset(const Pose& q,const Pose& mapping) {
    initial=home;kp=base_kp;kd=base_kd;validate();
    for(int i=0;i<12;++i) {
      if(!std::isfinite(q[i]) || !std::isfinite(mapping[i]) || (i%3!=2 && mapping[i]!=0.) || std::abs(q[i]-mapping[i]-home[i])>.03)
        throw std::runtime_error("Confirmed point-up start required; no automatic approach");
      if(std::min(home[i],goal[i]-.2)+mapping[i]<inverted_triangle::low[i] || std::max(home[i],goal[i]+.2)+mapping[i]>inverted_triangle::high[i])
        throw std::runtime_error("Mapped roll/support envelope exceeds limits");
      entry[i]=q[i]-mapping[i];
    }
    offset=mapping;command=entry;velocity.fill(0);support=Support{};run_steps=completed=segment=0;
    ramp=elapsed=gain=gate_wait=max_error=max_torque=0;state=inverted_triangle::RAMP;fault=inverted_triangle::NONE;
  }
  bool settled(const Pose& q,const Pose& qd) const {
    for(int i=0;i<12;++i)if(std::abs(qd[i])>.1 || std::abs(q[i]-offset[i]-command[i])>.1)return false;
    return true;
  }
  void step(double dt,const Pose& encoder,const Pose& qd,const V3& gravity,int request=0,bool fresh=false) {
    using namespace inverted_triangle;
    if(state==FAULT)return;
    if(!std::isfinite(dt)||dt<=0||dt>.01){stop(TIMING);return;}
    if(!gravity.allFinite()||std::abs(gravity.norm()-1)>.02){stop(SENSORS);return;}
    if(-gravity[2]<=std::cos(tilt_limit)){stop(TILT);return;}
    if(fresh&&(request<0||request>1)){stop(request==-1?OPERATOR:COMMAND);return;}
    Pose q;max_error=0;max_torque=0;
    for(int i=0;i<12;++i) {
      if(!std::isfinite(encoder[i])||!std::isfinite(qd[i])){stop(SENSORS);return;}
      q[i]=encoder[i]-offset[i];double error=std::abs(command[i]-q[i]);max_error=std::max(max_error,error);
      if(error>(i%3==2?.15:.25)){stop(TRACKING);return;}
      if(std::abs(qd[i])>2.){stop(SPEED);return;}
      if(encoder[i]<low[i]||encoder[i]>high[i]){stop(TRACKING);return;}
    }
    if(state==RAMP) {
      ramp=std::min(2.,ramp+dt);gain=smooth(ramp/2.);
      for(int i=0;i<12;++i)command[i]=entry[i]+gain*(home[i]-entry[i]);
      if(ramp>=2.)state=READY;
    } else if(state==READY && fresh && request==1 && settled(encoder,qd)) {
      state=RUNNING;run_steps=0;
    }
    if(state==RUNNING) {
      int k=run_steps++;elapsed=run_steps/hz;
      segment=k<1040?0:(k<7280?1:2);
      for(int i=0;i<12;++i)kd[i]=base_kd[i]*(k<7280?2.:1.);
      Pose target=home;
      if(k>=1040 && k<7280)for(int i=0;i<12;++i)target[i]=home[i]+smooth((k-1039.)/6240.)*(goal[i]-home[i]);
      if(k>=7280 && !stand_only) {
        if(k%10==0)support.update(q,qd,command,gravity,10./hz);
        for(int i=0;i<12;++i) {
          double speed=i%3==2?.5:(i%3==0?.45:.65),accel=i%3==2?1.2:2.;
          double desired=std::clamp(5*(goal[i]+support.offset[i]-command[i]),-speed,speed);
          double next=velocity[i]+std::clamp(desired-velocity[i],-accel/hz,accel/hz);
          target[i]=std::clamp(command[i]+next/hz,low[i]-offset[i],high[i]-offset[i]);
        }
      }
      for(int i=0;i<12;++i){velocity[i]=(target[i]-command[i])*hz;command[i]=target[i];}
      if(run_steps>=(stand_only?7280:23920)){
        state=DONE;completed=1;segment=3;
        if(stand_only)kd=base_kd;
      }
    }
    for(int i=0;i<12;++i) {
      double torque=gain*(kp[i]*(command[i]-q[i])-kd[i]*qd[i]);max_torque=std::max(max_torque,std::abs(torque));
      if(!std::isfinite(torque)||std::abs(torque)>1.5){stop(TORQUE);return;}
    }
  }
};
}
