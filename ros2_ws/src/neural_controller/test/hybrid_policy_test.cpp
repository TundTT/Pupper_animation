#include <RTNeural/RTNeural.h>
#include "neural_controller/wheel_align_hybrid.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using Hybrid = neural_controller::WheelAlignHybrid;
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double a, double b, const char *message) { require(std::abs(a - b) < 1e-9, message); }

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: hybrid_policy_test POLICY_JSON REFERENCE_CSV");
    std::ifstream stream(argv[1]);
    auto model = RTNeural::json_parser::parseJson<float>(stream, false);
    require(model && model->getInSize() == 51 && model->getOutSize() == 8, "network dimensions");
    std::ifstream fixtures(argv[2]);
    std::string line;
    int count = 0;
    double max_error = 0;
    while (std::getline(fixtures, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      std::array<float, 51> input{};
      std::array<float, 8> expected{};
      for (auto &x : input) require(bool(row >> x), "missing observation");
      for (auto &x : expected) require(bool(row >> x), "missing expected action");
      model->forward(input.data());
      for (int i = 0; i < 8; ++i) {
        double error = std::abs(double(model->getOutputs()[i]) - expected[i]);
        require(std::isfinite(error) && error < 3e-5, "RTNeural/Brax inference mismatch");
        max_error = std::max(error, max_error);
      }
      ++count;
    }
    require(count == 128, "inference fixture count");
    const double pi = std::acos(-1.0);
    near(Hybrid::wheel_pd(-pi + .1, pi - .1, .2), .33, "PD wrapping/damping");
    near(Hybrid::wheel_pd(2, 0, 0), 2, "PD positive saturation");
    near(Hybrid::wheel_pd(-2, 0, 0), -2, "PD negative saturation");
    std::array<double, 12> q{1, 0, .3, -1, 0, -.4, 1, 0, .5, -1, 0, -.6}, qd{};
    const auto initial = q;
    Hybrid h;
    h.recalibrate_home(q);
    h.reset(q);
    for (int k = 0; k < 4; ++k) {
      near(h.target[k], Hybrid::wrap(q[3*k+2]+pi), "activation calibration");
      q[3*k+2] += .1;
    }
    for (double cmd : h.wheel_commands(q, qd)) near(cmd, -.2, "all idle wheels actively hold");
    for (int command = 1; command <= 4; ++command) {
      q = initial;
      qd.fill(0);
      h.reset(q);
      h.select_command(command, q);
      int k = Hybrid::command_leg[command], hip = 3*k+1, wheel = 3*k+2;
      require(h.leg() == k && h.effective_command() == command, "leg index mapping");
      q[hip] = k % 2 == 0 ? 1.2 : -1.2;
      double default_abd = initial[3*k];
      require(h.lift_ready(q, default_abd, {0, 0, 0}, -1), "signed hip gate");
      require(!h.lift_ready(q, default_abd, {.3, 0, 0}, -1), "angular speed strict gate");
      require(!h.lift_ready(q, default_abd, {0, 0, 0}, -std::cos(.12)), "tilt strict gate");
      q[3*k] += .36;
      require(!h.lift_ready(q, default_abd, {0, 0, 0}, -1), "abduction gate");
      q[3*k] = default_abd;
      for (int n = 0; n < 9; ++n) { h.begin_step(q, true, .02); h.finish_step(q, qd); }
      require(h.phase == Hybrid::LIFT, "gate must dwell ten consecutive steps");
      h.begin_step(q, false, .02); h.finish_step(q, qd);
      require(h.gate_steps == 0, "lost gate resets dwell");
      for (int n = 0; n < 10; ++n) { h.begin_step(q, true, .02); h.finish_step(q, qd); }
      require(h.phase == Hybrid::ROTATE, "lift to rotate");
      double ref = h.reference[k];
      h.begin_step(q, true, .02);
      near(std::abs(Hybrid::wrap(h.reference[k] - ref)), .005, "slew 0.25 rad/s");
      h.finish_step(q, qd);
      q[wheel] += .2;
      h.begin_step(q, false, .02);
      const double snapshot = h.reference[k];
      near(snapshot, Hybrid::wrap(q[wheel]), "gate loss latches measured angle");
      h.finish_step(q, qd);
      q[wheel] += .1;
      h.begin_step(q, false, .02);
      near(h.reference[k], snapshot, "pause holds frozen snapshot, not drifting encoder");
      near(h.wheel_commands(q, qd)[k], -.2, "paused wheel actively holds");
      h.finish_step(q, qd);
      h.begin_step(q, true, .04);
      near(std::abs(Hybrid::wrap(h.reference[k] - snapshot)), .01, "resume uses actual dt");
      q[wheel] = h.target[k];
      h.finish_step(q, qd);
      require(h.phase == Hybrid::VERIFY, "rotate to verify");
      qd[wheel] = .08;
      h.begin_step(q, true, .02); h.finish_step(q, qd);
      require(h.settled_steps == 0, "strict speed threshold resets settling");
      qd[wheel] = 0;
      for (int n = 0; n < 24; ++n) { h.begin_step(q, true, .02); h.finish_step(q, qd); }
      require(h.phase == Hybrid::VERIFY, "settle dwell 25");
      h.begin_step(q, true, .02); h.finish_step(q, qd);
      require(h.phase == Hybrid::LOWER && h.verified && h.effective_command() == 0, "verified lowering commands stand");
      near(h.hold[k], Hybrid::wrap(q[wheel]), "latch hold before lowering per source");
      q[hip] = 0;
      for (int n = 0; n < 49; ++n) { h.begin_step(q, false, .02); h.finish_step(q, qd); }
      require(h.phase == Hybrid::LOWER, "lower dwell 50");
      qd[hip] = .2;
      h.begin_step(q, false, .02); h.finish_step(q, qd);
      require(h.phase == Hybrid::LOWER, "lower waits for hip speed");
      qd[hip] = 0;
      h.begin_step(q, false, .02); h.finish_step(q, qd);
      require(h.phase == Hybrid::HOLD && h.completed[k], "touchdown marks completed");
      h.select_command(command, q);
      require(h.phase == Hybrid::HOLD, "completed leg is not retried in same session");
    }
    // Every mid-sequence phase can be interrupted, including a switch to stand.
    for (auto phase : {Hybrid::LIFT, Hybrid::ROTATE, Hybrid::VERIFY}) {
      for (int requested : {0, 2}) {
        q = initial;
        h.reset(q); h.select_command(1, q); h.phase = phase;
        q[5] += .7;
        h.select_command(requested, q);
        require(h.phase == Hybrid::LOWER && !h.verified && h.effective_command() == 0, "interrupt lowers immediately");
        near(h.hold[1], Hybrid::wrap(q[5]), "interrupt freezes hold");
        for (int n = 0; n < 50; ++n) { h.begin_step(q, false, .02); h.finish_step(q, qd); h.select_command(requested, q); }
        require(!h.completed[1], "interrupted leg not completed");
        require(requested == 0 ? h.phase == Hybrid::HOLD : h.phase == Hybrid::LIFT && h.leg() == 0,
                "pending command only starts after lowering");
      }
    }
    h.reset(initial);
    require(h.phase == Hybrid::IDLE && !h.completed[0] && h.command == 0 && !h.step_pending,
            "reactivation clears phase/completion state");
    std::cout << "PASS: hybrid gate/PD/phase/reset cases and " << count
              << " Brax/RTNeural cases; max action error=" << max_error << '\n';
    return 0;
  } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
