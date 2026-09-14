#include "control_board_hardware_interface/stanford_stop_homing.hpp"
#include <cassert>
#include <limits>
using stanford_stop_homing::Joint;
int main() {
  // A fixed endpoint produces the original filtered torque threshold behavior.
  Joint stopped(-2.13,-1,-1.22);
  bool found=false;
  for(int i=0;i<200 && !found;++i) {
    stopped.observe(-2.13,0);
    found=stopped.seek(0,.01);
    assert(std::abs((stopped.target-stopped.position)*Joint::kp+stopped.velocity*Joint::kd)<=2.500001);
  }
  assert(found && stopped.raw_stop==-2.13);
  // A freely tracking joint must not be mistaken for a stop.
  Joint free(0,1,.42);
  for(int i=0;i<100;++i) {
    free.observe(free.target,free.velocity);
    assert(!free.seek(free.velocity,.01));
  }
  // Return is based on observed position, not merely the commanded target.
  for(int i=0;i<200;++i) assert(!stopped.return_to_zero(0,.01));
  stopped.observe(stopped.raw_stop-stopped.model_stop,0);
  assert(stopped.return_to_zero(0,.01));
  // A nonzero saved home must be reached in the stop-derived frame.
  const double home=1.1008971066674806;
  assert(!stopped.return_to_home(0,.01,home));
  stopped.observe(stopped.raw_stop-stopped.model_stop+home,0);
  assert(stopped.return_to_home(0,.01,home));
  for(double bad : {std::numeric_limits<double>::quiet_NaN(),10.}) {
    bool rejected=false;
    try { free.observe(bad,0); } catch(...) { rejected=true; }
    assert(rejected);
  }
}
