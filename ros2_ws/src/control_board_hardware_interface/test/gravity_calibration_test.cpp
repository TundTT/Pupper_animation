#include "control_board_hardware_interface/gravity_calibration.hpp"
#include <iostream>
#include <limits>

void require(bool pass, const char* what) { if (!pass) throw std::runtime_error(what); }
int main() {
  using namespace gravity_calibration;
  Sample sample; Pose q{}, reference{};
  for (size_t i = 0; i < q.size(); ++i) { q[i] = 7. + i; reference[i] = .2 * i; }
  bool ready = false;
  for (int t = 0; t < 110; ++t) ready = sample.observe(q, t * .01, true);
  require(ready, "stationary capture");
  auto offset = sample.offsets(reference);
  for (size_t i = 0; i < q.size(); ++i)
    require(std::abs(q[i] - offset[i] - reference[i]) < 1e-12, "unwrapped calibrated coordinates");
  require(!sample.observe(q, 1.1, false), "bad transport must reset even with repeated positions");
  bool rejected = false;
  try { sample.offsets(reference); } catch (...) { rejected = true; }
  require(rejected, "cannot use old sample after transport loss");
  for (int t = 0; t < 300; ++t) {
    q[0] += .001;
    require(!sample.observe(q, 2. + t * .01, true), "moving joint cannot calibrate");
  }
  sample.reset();
  for (int t = 0; t < 90; ++t) sample.observe(q, 6. + t*.01, true);
  require(!sample.observe(q, 7.5, true), "gap cannot complete window");
  q[4] = std::numeric_limits<double>::quiet_NaN();
  require(!sample.observe(q, 7.51, true), "nonfinite input");
  std::cout << "PASS gravity pose window, winding, motion, gaps and bad feedback\n";
}
