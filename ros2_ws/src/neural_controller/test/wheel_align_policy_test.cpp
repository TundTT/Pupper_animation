// Standalone RTNeural inference parity; this does not activate a ROS controller.
#include <RTNeural/RTNeural.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: wheel_align_policy_test POLICY_JSON BRAX_REFERENCE_CSV");
    std::ifstream stream(argv[1]);
    nlohmann::json policy;
    stream >> policy;
    require(policy.at("behavior") == "wheel_align", "wrong behavior");
    require(policy.at("observation_history") == 4, "wrong history length");
    require(policy.at("single_observation_size") == 94, "wrong frame size");
    require(policy.at("policy_action_size") == 3, "wrong correction count");
    require(!policy.contains("action_scale") && !policy.contains("action_types"),
            "corrections must not masquerade as direct joint actions");
    int offset = 0;
    for (const auto &block : policy.at("observation_layout")) {
      require(block.at("offset") == offset, "noncontiguous observation layout");
      offset += block.at("size").get<int>();
    }
    require(offset == 94, "incomplete observation contract");
    std::ifstream model_stream(argv[1]);
    auto model = RTNeural::json_parser::parseJson<float>(model_stream, false);
    require(model && model->getInSize() == 376 && model->getOutSize() == 3,
            "network dimensions mismatch");
    std::ifstream fixtures(argv[2]);
    std::string line;
    int count = 0;
    double max_error = 0;
    while (std::getline(fixtures, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      std::vector<float> input(376), expected(3);
      for (auto &x : input) require(bool(row >> x), "missing observation column");
      for (auto &x : expected) require(bool(row >> x), "missing action column");
      std::string extra;
      require(!(row >> extra), "extra fixture column");
      model->forward(input.data());
      for (int i = 0; i < 3; ++i) {
        const double error = std::abs(double(model->getOutputs()[i]) - expected[i]);
        require(std::isfinite(error) && error < 3e-4, "RTNeural/Brax action mismatch");
        max_error = std::max(max_error, error);
      }
      ++count;
    }
    require(count >= 512, "missing inference fixtures");
    std::cout << "PASS: wheel-align contract and " << count
              << " Brax/RTNeural cases; max action error=" << max_error << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
