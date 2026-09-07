// Standalone checks using the same RTNeural Eigen backend as the ROS controller.
#include <RTNeural/RTNeural.h>
#include "neural_controller/policy_contract.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: walk_policy_test POLICY_JSON REFERENCE_CSV");
    std::ifstream stream(argv[1]);
    nlohmann::json policy;
    stream >> policy;
    neural_controller::PolicyContract contract(policy);
    require(contract.command({-0.75, 0.5, -2.0}) ==
                std::array<double, 3>{-0.35, 0.15, -0.8}, "command envelope mismatch");
    require(contract.command({0.2, -0.1, 0.5}) ==
                std::array<double, 3>{0.2, -0.1, 0.5}, "in-range command changed");
    require(contract.command({std::numeric_limits<double>::infinity(), 0.1, 0.5}) ==
                std::array<double, 3>{0, 0, 0}, "nonfinite command not stopped");
    require(contract.fixed_orientation && contract.orientation ==
                std::array<double, 3>{0, 0, 1}, "upright contract mismatch");
    neural_controller::PolicyContract legacy;
    require(!legacy.fixed_orientation && legacy.command({0.75, 0.5, 2.0}) ==
                std::array<double, 3>{0.75, 0.5, 2.0}, "legacy command behavior changed");
    for (const auto &bad : std::vector<nlohmann::json>{
             {{"command_low", {-0.1, -0.1, -0.1}}},
             {{"command_low", {0.1, -0.1, -0.1}}, {"command_high", {1, 1, 1}}},
             {{"orientation_command", {0, 0, 0}}}}) {
      bool rejected = false;
      try { neural_controller::PolicyContract invalid(bad); }
      catch (const std::exception &) { rejected = true; }
      require(rejected, "invalid contract accepted");
    }
    std::vector<float> observation(144, 123.0f);
    observation.assign(144, 0.0f);
    for (int i = 0; i < 24; ++i) observation[i] = float(i) / 24;
    neural_controller::seed_observation_history(observation, 36);
    for (int i = 0; i < 144; ++i) {
      require(observation[i] == observation[i % 36], "history seed mismatch");
      if (i % 36 >= 24) require(observation[i] == 0, "stale action in reset history");
    }

    std::ifstream model_stream(argv[1]);
    auto model = RTNeural::json_parser::parseJson<float>(model_stream, false);
    require(model && model->getInSize() == 144 && model->getOutSize() == 12,
            "network dimensions mismatch");
    std::ifstream fixtures(argv[2]);
    std::string line;
    int count = 0;
    double max_error = 0;
    while (std::getline(fixtures, line)) {
      if (line.empty() || line.front() == '#') continue;
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      std::vector<float> input(144), expected(12);
      for (auto &x : input) require(bool(row >> x), "missing observation column");
      for (auto &x : expected) require(bool(row >> x), "missing expected action column");
      model->forward(input.data());
      for (int i = 0; i < 12; ++i) {
        double error = std::abs(double(model->getOutputs()[i]) - expected[i]);
        require(std::isfinite(error) && error < 3e-5, "RTNeural/JAX action mismatch");
        max_error = std::max(max_error, error);
      }
      ++count;
    }
    require(count >= 64, "not enough inference fixtures");
    std::cout << "PASS: command contract, reset history, and " << count
              << " JAX/RTNeural observations; max action error=" << max_error << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
