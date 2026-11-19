#include "wheel_align_spin.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
int main() {
  WheelAlignSpin controller;
  std::array<double,4> q{}, speed{}, goal{1.,2.,-1.,-2.}, enable{1.,1.,1.,1.};
  controller.reset(q);
  for(int tick=0;tick<2;++tick) {
    const auto target=controller.apply(q,speed,goal,enable,0);
    assert(target==q);
  }
  auto target=controller.apply(q,speed,goal,enable,0);
  assert(std::abs(target[0]-.06)<1e-12);
  for(int leg=1;leg<4;++leg) assert(target[leg]==q[leg]);
  q[0]=1.;
  for(int tick=0;tick<5;++tick) controller.apply(q,speed,goal,enable,0);
  assert(controller.hold.held[0]);
  q[0]+=20.*3.141592653589793+.2;
  target=controller.apply(q,speed,goal,enable,1);
  assert(std::abs(target[0]-(1.+20.*3.141592653589793))<1e-12);
  enable.fill(-1.);
  q[1]=.15;
  target=controller.apply(q,speed,goal,enable,1);
  assert(std::abs(target[1]-.15)<1e-12);
  controller.reset(q);
  controller.reference[1]=goal[1]-.02;
  controller.enable_ticks[1]=3;
  q[1]=goal[1]-.4; speed[1]=1.; enable[1]=1.;
  controller.apply(q,speed,goal,enable,1);
  assert(controller.goal_commanded[1] && !controller.hold.held[1]);
  enable[1]=-1.;
  target=controller.apply(q,speed,goal,enable,1);
  assert(std::abs(target[1]-goal[1])<1e-12);
}
