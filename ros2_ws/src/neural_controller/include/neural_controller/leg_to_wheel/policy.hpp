#pragma once
#include <RTNeural/RTNeural.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace leg_to_wheel {
using Joints = std::array<double, 12>;
using Vec3 = std::array<double, 3>;
inline constexpr std::array<int,5> command_to_foot{-1,1,0,2,3};
inline constexpr std::array<double,4> lift_sign{1.,-1.,1.,-1.};

// Operator verification replaces the clearance/contact sequencer for the manual
// adapter. A lower request is NOT evidence of contact or successful conversion.
struct ManualSequencer {
  int step=0;
  int command() const {return step%2 ? (step+1)/2 : 0;}
  int leg() const {return step ? (step+1)/2 : 0;}
  bool lower_requested() const {return step>0 && step%2==0;}
  bool request(int next) {
    if(next!=step+1 || next>8)return false;
    step=next;return true;
  }
};

struct Sequencer {
  enum class Phase {stand, lifting, heating, lowering};
  int command=0, active=0, converted=0;
  Phase phase=Phase::stand;
  double stable_seconds=0.;
  double clearance_gate=.015, settle_seconds=.3;

  bool request(int next) {
    if(next<1 || next>4)throw std::invalid_argument("Expected command FL/FR/BR/BL (1..4)");
    if(phase!=Phase::stand || (converted & (1<<command_to_foot[next])))return false;
    command=active=next;phase=Phase::lifting;stable_seconds=0.;return true;
  }
  int update(double dt,double clearance,bool contact,bool heating_confirmed=false) {
    if(!std::isfinite(dt) || dt<0 || !std::isfinite(clearance))throw std::invalid_argument("Invalid sequencing measurement/time");
    if(phase==Phase::lifting || phase==Phase::heating) {
      bool unloaded=clearance>=clearance_gate && !contact;
      stable_seconds=unloaded?stable_seconds+dt:0.;
      phase=stable_seconds>=settle_seconds?Phase::heating:Phase::lifting;
      if(phase==Phase::heating && heating_confirmed) {command=0;phase=Phase::lowering;stable_seconds=0.;}
    } else if(phase==Phase::lowering) {
      stable_seconds=contact?stable_seconds+dt:0.;
      if(stable_seconds>=settle_seconds) {converted|=1<<command_to_foot[active];active=0;phase=Phase::stand;stable_seconds=0.;}
    }
    return command;
  }
};

// Actor runtime: calibrated model-frame joints and body-frame IMU are required.
// step() retains the original measured-clearance filter; step_manual() is the
// operator-verified time-only variant. Neither function estimates contact.
class Policy {
 public:
  struct Output {Joints raw_action{}, applied_action{}, position_target{};};
  explicit Policy(const std::string& path) {
    std::ifstream stream(path); nlohmann::json j;stream>>j;
    if(j.at("behavior")!="leg_to_wheel" || j.at("observation_history")!=4 || j.at("single_observation_size")!=35 ||
       j.at("command_states")!=nlohmann::json({"stand","FL","FR","BR","BL"}) ||
       j.at("lowering").at("filter")!="hip_descent_v1")throw std::runtime_error("Leg-to-wheel policy contract mismatch");
    home=j.at("default_joint_pos").get<Joints>();scale=j.at("action_scale").get<Joints>();
    lower=j.at("joint_lower_limits").get<Joints>();upper=j.at("joint_upper_limits").get<Joints>();
    kp=j.at("kps").get<Joints>();kd=j.at("kds").get<Joints>();
    dt_=j.at("ctrl_dt").get<double>();speed_=j.at("lowering").at("initial_hip_speed_rad_s").get<double>();
    ease_=j.at("lowering").at("ease_seconds").get<double>();
    fade_=j.at("lowering").at("clearance_fade_m").get<std::array<double,2>>();
    for(int i=0;i<12;++i)if(!std::isfinite(home[i]) || !std::isfinite(scale[i]) || scale[i]<=0 ||
      !std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>upper[i] ||
      !std::isfinite(kp[i]) || !std::isfinite(kd[i]) || kp[i]<0 || kd[i]<0)throw std::runtime_error("Invalid actuator contract");
    if(!std::isfinite(dt_) || dt_<=0 || !std::isfinite(speed_) || speed_<0 || !std::isfinite(ease_) || ease_<0 ||
       !std::isfinite(fade_[0]) || !std::isfinite(fade_[1]) || fade_[1]<=fade_[0])throw std::runtime_error("Invalid timing/clearance contract");
    model_=RTNeural::json_parser::parseJson<float>(j,false);
    if(!model_ || model_->getInSize()!=140 || model_->getOutSize()!=12)throw std::runtime_error("Network shape mismatch");
    reset();
  }
  void reset() {seed_=true;previous_raw_.fill(0);previous_target_.fill(0);observation.fill(0);previous_command_=0;lowering_foot_=-1;elapsed_=0.;}
  Output step(const Vec3& omega,const Vec3& gravity,const Joints& q,int command,
              const std::array<double,4>& clearance,double dt=.02) {
    for(double x:clearance)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite capsule clearance");
    return step_impl(omega,gravity,q,command,&clearance,dt);
  }
  // Explicit manual variant: time easing only, without invented floor readings.
  // The actor, raw-action history and target limits retain the export contract.
  Output step_manual(const Vec3& omega,const Vec3& gravity,const Joints& q,int command,double dt=.02) {
    return step_impl(omega,gravity,q,command,nullptr,dt);
  }
 private:
  Output step_impl(const Vec3& omega,const Vec3& gravity,const Joints& q,int command,
                   const std::array<double,4>* clearance,double dt) {
    if(command<0 || command>4 || !std::isfinite(dt) || std::abs(dt-dt_)>1e-7)throw std::invalid_argument("Expected one command at the trained control timestep");
    for(double x:omega)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite gyro");
    for(double x:gravity)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite gravity");
    for(double x:q)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite joint position");
    if(command!=0) {lowering_foot_=-1;elapsed_=0.;}
    else if(previous_command_!=0) {lowering_foot_=command_to_foot[previous_command_];elapsed_=0.;}
    std::array<float,35> frame{};
    for(int i=0;i<3;++i) {frame[i]=float(omega[i]);frame[3+i]=float(gravity[i]);}
    frame[6+command]=1.f;
    for(int i=0;i<12;++i) {frame[11+i]=float(q[i]-home[i]);frame[23+i]=float(previous_raw_[i]);}
    if(seed_) {for(int h=0;h<4;++h)std::copy(frame.begin(),frame.end(),observation.begin()+35*h);seed_=false;}
    else {std::move_backward(observation.begin(),observation.end()-35,observation.end());std::copy(frame.begin(),frame.end(),observation.begin());}
    model_->forward(observation.data()); Output out;
    for(int i=0;i<12;++i) {
      double raw=model_->getOutputs()[i];if(!std::isfinite(raw))throw std::runtime_error("Nonfinite policy output");
      out.raw_action[i]=out.applied_action[i]=std::clamp(raw,-1.,1.);
    }
    if(lowering_foot_>=0 && speed_>0) {
      double phase=ease_>0?std::min(elapsed_/ease_,1.):0.;
      double strength=(1.-3.*phase*phase+2.*phase*phase*phase);
      if(clearance)strength*=std::clamp(((*clearance)[lowering_foot_]-fade_[0])/(fade_[1]-fade_[0]),0.,1.);
      int joint=3*lowering_foot_+1;double sign=lift_sign[lowering_foot_],delta=speed_*dt/scale[joint];
      if((out.applied_action[joint]-previous_target_[joint])*sign < -delta)
        out.applied_action[joint]+=strength*(previous_target_[joint]-sign*delta-out.applied_action[joint]);
      elapsed_+=dt;
    }
    for(int i=0;i<12;++i)out.position_target[i]=std::clamp(home[i]+out.applied_action[i]*scale[i],lower[i],upper[i]);
    previous_raw_=out.raw_action;previous_target_=out.applied_action;previous_command_=command;
    return out;
  }
 public:
  Joints home{},scale{},lower{},upper{},kp{},kd{};
  std::array<float,140> observation{};
 private:
  std::unique_ptr<RTNeural::Model<float>> model_;
  Joints previous_raw_{},previous_target_{};
  bool seed_=true;
  int previous_command_=0,lowering_foot_=-1;
  double elapsed_=0.,dt_=.02,speed_=8.,ease_=.2;
  std::array<double,2> fade_{.02,.04};
};
}  // namespace leg_to_wheel
