#include "wheel_align_hold.h"
#include <cassert>
int main() {
  WheelAlignHold controller;
  std::array<double,4> goal{3.141592653589793,1.,-2.,4.};
  auto q=goal;
  std::array<double,4> speed{}, learned{0.,0.,0.,0.};
  for(int tick=0;tick<4;++tick) assert(controller.apply(q,speed,goal,learned,0)[0]==0.);
  assert(controller.apply(q,speed,goal,learned,0)[0]==goal[0]);
  q[0] += 20.*3.141592653589793+.2;
  const auto hold=controller.apply(q,speed,goal,learned,1);
  assert(std::abs(hold[0]-(goal[0]+20.*3.141592653589793))<1e-12);
  speed[1]=1.;
  for(int tick=0;tick<10;++tick) controller.apply(q,speed,goal,learned,1);
  assert(!controller.held[1]);
  controller.reset();
  for(bool held:controller.held) assert(!held);
}
