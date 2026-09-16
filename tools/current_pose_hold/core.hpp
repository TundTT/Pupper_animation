#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace measurement_hold {
using Pose=std::array<double,12>;
struct Core {
  Pose target{};double elapsed=0,gain=0,error=0;bool stopped=true;
  void reset(const Pose& q,const Pose& v) {
    for(int i=0;i<12;++i)if(!std::isfinite(q[i])||!std::isfinite(v[i])||std::abs(v[i])>.1)
      throw std::runtime_error("Finite stationary feedback required");
    target=q;elapsed=gain=error=0;stopped=false;
  }
  void stop(){stopped=true;gain=0;}
  void step(double dt,const Pose&q,const Pose&v,bool joy_ok) {
    if(stopped)return;
    if(!joy_ok||!std::isfinite(dt)||dt<=0||dt>.01){stop();return;}
    error=0;
    for(int i=0;i<12;++i) {
      error=std::max(error,std::abs(target[i]-q[i]));
      const double effort=gain*(7.5*(target[i]-q[i])-.35*v[i]);
      if(!std::isfinite(q[i])||!std::isfinite(v[i])||std::abs(v[i])>2||error>.25||std::abs(effort)>3){stop();return;}
    }
    elapsed+=dt;double u=std::clamp(elapsed/3.,0.,1.);gain=u*u*(3-2*u);
  }
};
}
