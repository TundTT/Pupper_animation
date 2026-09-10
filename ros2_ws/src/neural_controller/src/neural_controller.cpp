#include "neural_controller/neural_controller.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace neural_controller {
NeuralController::NeuralController()
    : controller_interface::ControllerInterface(),
      rt_cmd_vel_ptr_(nullptr),
      rt_cmd_pose_ptr_(nullptr),
      rt_leg_lift_command_ptr_(nullptr) {}

// Check parameter vectors have the correct size
bool NeuralController::check_param_vector_size() {
  const std::vector<std::pair<std::string, size_t>> param_sizes = {
      {"action_scales", params_.action_scales.size()},
      {"action_types", params_.action_types.size()},
      {"kps", params_.kps.size()},
      {"kds", params_.kds.size()},
      {"init_kps", params_.init_kps.size()},
      {"init_kds", params_.init_kds.size()},
      {"default_joint_pos", params_.default_joint_pos.size()},
      {"joint_lower_limits", params_.joint_lower_limits.size()},
      {"joint_upper_limits", params_.joint_upper_limits.size()},
      {"joint_names", params_.joint_names.size()}};

  for (const auto &[name, size] : param_sizes) {
    if (size != kActionSize) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s size is %ld, expected %d", name.c_str(), size,
                   kActionSize);
      return false;
    }
  }
  return true;
}

controller_interface::CallbackReturn NeuralController::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();

    if (params_.gain_multiplier < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gain_multiplier must be >= 0.0. Stopping");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (params_.gain_multiplier != 1.0) {
      RCLCPP_WARN(get_node()->get_logger(), "Gain_multiplier is set to %f",
                  params_.gain_multiplier);
    }

    std::ifstream json_stream(params_.model_path, std::ifstream::binary);
    model_ = RTNeural::json_parser::parseJson<float>(json_stream, true);

    // Read params json file using nholsojson to extract metadata
    nlohmann::json j;
    std::ifstream json_file(params_.model_path);
    json_file >> j;

    policy_contract_ = PolicyContract(j);
    if (j.contains("joint_names") &&
        j.at("joint_names").get<std::vector<std::string>>() != params_.joint_names) {
      throw std::runtime_error("Configured joint order differs from the policy export");
    }
    if (j.contains("action_types") &&
        j.at("action_types").get<std::vector<std::string>>() != params_.action_types) {
      throw std::runtime_error("Configured action types differ from the policy export");
    }
    behavior_ = j.value("behavior", std::string("locomotion"));
    policy_action_size_ = behavior_ == "wheel_align_hybrid" ? 8 : kActionSize;
    if (!model_ || model_->layers.empty() || model_->getOutSize() != policy_action_size_) {
      throw std::runtime_error("Policy output size differs from its behavior contract");
    }
    if (policy_contract_.bounded_commands) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Policy command limits: vx [%g,%g], vy [%g,%g], yaw [%g,%g]",
                  policy_contract_.low[0], policy_contract_.high[0],
                  policy_contract_.low[1], policy_contract_.high[1],
                  policy_contract_.low[2], policy_contract_.high[2]);
    }

    auto set_param_from_json_vector = [&](const std::string &key, auto &param) {
      if (j.find(key) != j.end()) {
        RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting %s vector element-by-element",
                    key.c_str());
        if (j[key].size() != kActionSize) {
          std::string error_msg = "Invalid size for " + key + " (" + std::to_string(j[key].size()) +
                                  ") != " + std::to_string(kActionSize);
          RCLCPP_ERROR(get_node()->get_logger(), "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }
        param.resize(j[key].size(), 0.0);
        for (int i = 0; i < param.size(); i++) {
          param.at(i) = j[key].at(i);
        }
      }
    };

    auto set_param_from_json_scalar = [&](const std::string &key, auto &param, int size) {
      if (j.find(key) != j.end()) {
        RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting %s[:]=%f", key.c_str(),
                    static_cast<double>(j[key]));
        param.resize(size, 0.0);
        for (auto &p : param) {
          p = j[key];
        }
      }
    };

    auto set_param_from_json_mixed = [&](const std::string &key, auto &param, int size) {
      if (j.find(key) != j.end()) {
        if (j[key].is_array()) {
          set_param_from_json_vector(key, param);
        } else {
          set_param_from_json_scalar(key, param, size);
        }
      }
    };

    set_param_from_json_scalar("kp", params_.kps, kActionSize);
    set_param_from_json_scalar("kd", params_.kds, kActionSize);
    set_param_from_json_mixed("action_scale", params_.action_scales, kActionSize);
    set_param_from_json_vector("default_joint_pos", params_.default_joint_pos);
    set_param_from_json_vector("joint_lower_limits", params_.joint_lower_limits);
    set_param_from_json_vector("joint_upper_limits", params_.joint_upper_limits);

    // Warn user that use_imu should be set in the robot description
    if (j.find("use_imu") != j.end()) {
      params_.use_imu = j["use_imu"];
      RCLCPP_WARN(get_node()->get_logger(),
                  "From JSON, setting params_use_imu=%d. Verify robot description has proper value "
                  "of use_imu too.",
                  params_.use_imu);
    }

    if (j.find("observation_history") != j.end()) {
      params_.observation_history = j["observation_history"];
      RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting params_.observation_history=%ld",
                  params_.observation_history);
    }

    // Determine the observation layout from the policy's declared behavior. Policies exported
    // before "behavior" existed (the locomotion policies) have no such key -- default to
    // "locomotion" for them. Any other value is a hard error rather than a silent guess.
    behavior_ = j.find("behavior") != j.end() ? j["behavior"].get<std::string>() : "locomotion";
    if (behavior_ == "locomotion") {
      joint_position_idx_ = 3 + 3 + 3 + 3;  // ang_vel + gravity + xyyaw_vel_cmd + desired_world_z
      last_action_idx_ = joint_position_idx_ + kActionSize;
      single_observation_size_ = last_action_idx_ + kActionSize;
    } else if (behavior_ == "leg_lift") {
      if (j.find("command_states") == j.end()) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "behavior=\"leg_lift\" but model JSON has no \"command_states\"");
        return controller_interface::CallbackReturn::ERROR;
      }
      command_states_ = j["command_states"].get<std::vector<std::string>>();
      num_commands_ = static_cast<int>(command_states_.size());
      joint_position_idx_ = 3 + 3 + num_commands_;  // ang_vel + gravity + command_one_hot
      last_action_idx_ = joint_position_idx_ + kActionSize;
      single_observation_size_ = last_action_idx_ + kActionSize;
    } else if (behavior_ == "wheel_align_hybrid") {
      command_states_ = j.at("command_states").get<std::vector<std::string>>();
      num_commands_ = 5;
      joint_position_idx_ = 11;
      last_action_idx_ = 35;
      hybrid_.motion_version = j.value("motion_contract_version", 1);
      if (hybrid_.motion_version != 1 && hybrid_.motion_version != 2)
        throw std::runtime_error("Unsupported alignment motion version");
      if (hybrid_.motion_version == 2 && j.at("motion_contract_id") != "quadmorph-align-motion-v2")
        throw std::runtime_error("Alignment motion contract identifier mismatch");
      single_observation_size_ = hybrid_.motion_version == 2 ? 82 : 51;
      const std::vector<std::string> commands{"stand", "front_l", "front_r", "back_r", "back_l"};
      std::vector<std::string> blocks{"body_angular_velocity", "projected_gravity",
          "effective_command_one_hot", "joint_position", "joint_velocity", "last_action",
          "target_error_sin", "target_error_cos"};
      std::vector<int> sizes{3, 3, 5, 12, 12, 8, 4, 4};
      if (hybrid_.motion_version == 2) {
        for (const auto &block : {"phase", "progress", "motion_reference", "applied_position", "applied_velocity"}) blocks.push_back(block);
        for (int size : {6,1,8,8,8}) sizes.push_back(size);
        if (get_update_rate()!=520 || params_.repeat_action != 10 || j.at("ctrl_dt") != WheelAlignMotion::control_dt)
          throw std::runtime_error("Motion v2 requires 520 Hz manager / repeat 10");
        if (std::abs(params_.gain_multiplier-1.)>1e-6)
          throw std::runtime_error("Motion v2 requires the trained gain multiplier 1.0");
        for(int row=0;row<12;++row) {
          const bool wheel=row%3==2;
          if(std::abs(params_.kps[row]-(wheel ? 0. : 5.))>1e-6 ||
             std::abs(params_.kds[row]-(wheel ? .35 : .25))>1e-6)
            throw std::runtime_error("Motion v2 actuator gains differ from training");
        }
        for(int a=0;a<8;++a) {
          const int row=WheelAlignHybrid::position_rows[a];
          if (std::abs(params_.default_joint_pos[row]-WheelAlignMotion::neutral[a])>1e-6 ||
              std::abs(params_.joint_lower_limits[row]-WheelAlignMotion::low[a])>1e-6 ||
              std::abs(params_.joint_upper_limits[row]-WheelAlignMotion::high[a])>1e-6)
            throw std::runtime_error("Motion v2 pose/limit contract mismatch");
        }
      }
      int offset = 0;
      if (j.at("observation_layout").size() != blocks.size())
        throw std::runtime_error("Hybrid observation layout block count mismatch");
      for (size_t i = 0; i < blocks.size(); ++i) {
        const auto &block = j.at("observation_layout").at(i);
        if (block.at("name") != blocks[i] || block.at("offset") != offset || block.at("size") != sizes[i])
          throw std::runtime_error("Hybrid observation layout mismatch");
        offset += sizes[i];
      }
      if (command_states_ != commands || params_.observation_history != 1 ||
          j.at("position_joint_rows") != std::vector<int>({0, 1, 3, 4, 6, 7, 9, 10}) ||
          j.at("wheel_joint_rows") != std::vector<int>({2, 5, 8, 11}) ||
          j.at("command_leg") != std::vector<int>({-1, 1, 0, 2, 3}) ||
          j.at("single_observation_size") != single_observation_size_ || j.at("policy_action_size") != 8 ||
          !j.at("observation_clip").is_null() || params_.repeat_action < 1)
        throw std::runtime_error("Unsupported hybrid policy contract");
      for (int i = 0; i < kActionSize; ++i) {
        const bool wheel = i % 3 == 2;
        if (params_.action_types.at(i) != (wheel ? "velocity" : "position") ||
            (wheel && (params_.kps.at(i) != 0.0 || params_.init_kps.at(i) != 0.0)))
          throw std::runtime_error("Hybrid wheels require velocity mode with zero position gain");
      }
    } else if (behavior_ == "wheel") {
      // Velocity-command wheeled policy. Same command triple as locomotion, but no
      // desired_world_z, so the joint block starts 3 earlier (33 vs 36 per frame).
      joint_position_idx_ = 3 + 3 + 3;  // ang_vel + gravity + xyyaw_vel_cmd
      last_action_idx_ = joint_position_idx_ + kActionSize;
      single_observation_size_ = last_action_idx_ + kActionSize;

      for (const auto &key : {"wheel_joint_rows", "wheel_forward_sign",
                              "wheel_velocity_normalizer"}) {
        if (j.find(key) == j.end()) {
          RCLCPP_ERROR(get_node()->get_logger(),
                       "behavior=\"wheel\" but model JSON has no \"%s\"", key);
          return controller_interface::CallbackReturn::ERROR;
        }
      }
      wheel_joint_rows_ = j["wheel_joint_rows"].get<std::vector<int>>();
      wheel_forward_sign_ = j["wheel_forward_sign"].get<std::vector<double>>();
      wheel_velocity_normalizer_ = j["wheel_velocity_normalizer"];
      if (wheel_joint_rows_.size() != wheel_forward_sign_.size()) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "wheel_joint_rows (%zu) and wheel_forward_sign (%zu) differ in length",
                     wheel_joint_rows_.size(), wheel_forward_sign_.size());
        return controller_interface::CallbackReturn::ERROR;
      }
      if (wheel_velocity_normalizer_ == 0.0) {
        RCLCPP_ERROR(get_node()->get_logger(), "wheel_velocity_normalizer must be nonzero");
        return controller_interface::CallbackReturn::ERROR;
      }
      // Every declared wheel row must actually be a velocity-type joint, or the
      // observation and the action would disagree about what that row means.
      for (int row : wheel_joint_rows_) {
        if (row < 0 || row >= kActionSize) {
          RCLCPP_ERROR(get_node()->get_logger(), "wheel_joint_rows entry %d out of range", row);
          return controller_interface::CallbackReturn::ERROR;
        }
        if (params_.action_types.at(row) != "velocity") {
          RCLCPP_ERROR(get_node()->get_logger(),
                       "wheel row %d has action_type \"%s\", expected \"velocity\" -- check "
                       "config.yaml against the policy JSON",
                       row, params_.action_types.at(row).c_str());
          return controller_interface::CallbackReturn::ERROR;
        }
      }
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Unknown behavior \"%s\" in model JSON",
                   behavior_.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }

    // Check that the observation history is consistent with the model input shape
    if (j["in_shape"].at(1) != params_.observation_history * single_observation_size_) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "observation_history (%ld) * single_observation_size (%d) != in_shape (%d)",
                   params_.observation_history, single_observation_size_,
                   static_cast<int>(j["in_shape"].at(1)));
      return controller_interface::CallbackReturn::ERROR;
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!check_param_vector_size()) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  if (behavior_ == "wheel_align_hybrid") {
    startup_home_subscriber_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/wheel_align/startup_home", rclcpp::QoS(1).reliable().transient_local(),
        [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) { rt_startup_home_ptr_.writeFromNonRT(msg); });
  }
  RCLCPP_INFO(get_node()->get_logger(), "configure successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration NeuralController::command_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::InterfaceConfiguration NeuralController::state_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::CallbackReturn NeuralController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  // Clear command buffers to ignore pre-activation commands
  rt_cmd_vel_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>>(nullptr);
  rt_cmd_pose_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Pose>>(nullptr);

  // Populate the command interfaces map
  RCLCPP_INFO(get_node()->get_logger(), "Populating command interfaces map");
  command_interfaces_map_.clear();
  for (auto &command_interface : command_interfaces_) {
    RCLCPP_INFO(get_node()->get_logger(), "Prefix %s. Adding command interface %s",
                command_interface.get_prefix_name().c_str(),
                command_interface.get_interface_name().c_str());
    command_interfaces_map_[command_interface.get_prefix_name()].insert_or_assign(
        command_interface.get_interface_name(), std::ref(command_interface));
  }

  // Populate the state interfaces map
  state_interfaces_map_.clear();
  for (auto &state_interface : state_interfaces_) {
    RCLCPP_INFO(get_node()->get_logger(), "Prefix %s. Adding state interface %s",
                state_interface.get_prefix_name().c_str(),
                state_interface.get_interface_name().c_str());
    state_interfaces_map_[state_interface.get_prefix_name()].insert_or_assign(
        state_interface.get_interface_name(), std::ref(state_interface));
  }

  // Store the initial joint positions
  for (int i = 0; i < kActionSize; i++) {
    init_joint_pos_.at(i) =
        state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
  }

  if (behavior_ == "wheel_align_hybrid") {
    if (!std::all_of(init_joint_pos_.begin(), init_joint_pos_.end(),
                     [](double x) { return std::isfinite(x); }))
      return controller_interface::CallbackReturn::ERROR;
    auto startup = rt_startup_home_ptr_.readFromRT();
    if (!startup || !*startup || (*startup)->data.size() != 4 ||
        !std::all_of((*startup)->data.begin(), (*startup)->data.end(), [](double x) { return std::isfinite(x); })) {
      RCLCPP_ERROR(get_node()->get_logger(), "No valid startup wheel home. Alignment cannot activate; do not calibrate from the action button.");
      return controller_interface::CallbackReturn::ERROR;
    }
    auto home_q = init_joint_pos_;
    for (int k=0; k<4; ++k) home_q[3*k+2] = (*startup)->data[k];
    hybrid_.recalibrate_home(home_q);
    hybrid_.reset(init_joint_pos_);  // fresh holds; retains startup home
    motion_.reset(init_joint_pos_);
    hybrid_elapsed_ = 0.0;
    hybrid_first_step_ = true;
    for (auto &interface : command_interfaces_) interface.set_value(0.0);

  }

  // Reset estop caused by falling over
  estop_active_ = false;

  init_time_ = get_node()->now();
  repeat_action_counter_ = -1;

  cmd_x_vel_ = 0.0;
  cmd_y_vel_ = 0.0;
  cmd_yaw_vel_ = 0.0;

  // Initialize the observation vector
  observation_.assign(params_.observation_history * single_observation_size_, 0.0);
  seed_history_ = true;
  model_->reset();
  desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);

  // Set the gravity z-component in the initial observation vector
  for (int i = 0; i < params_.observation_history; i++) {
    observation_.at(i * single_observation_size_ + kGravityZIndx) = -1.0;
  }

  // "leg_lift" behavior: reset to "stand" (command_states_[0]) on every activation, and
  // (re)subscribe so a freshly-activated controller doesn't act on a stale buffer. QoS is
  // transient_local so a late-joining subscriber immediately gets the last command joy_util_node
  // published, instead of momentarily commanding "stand" until the next button press.
  // Hybrid uses its own volatile topic: publish after activation. It must not replay
  // a retained leg command against newly acquired provisional calibration.
  if (behavior_ == "leg_lift" || behavior_ == "wheel_align_hybrid") {
    command_index_ = 0;
    rt_leg_lift_command_ptr_ =
        realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::Int32>>(nullptr);
    leg_lift_command_subscriber_ = get_node()->create_subscription<std_msgs::msg::Int32>(
        behavior_ == "wheel_align_hybrid" ? "/wheel_align_hybrid_command_index" : "/leg_lift_command_index",
        behavior_ == "wheel_align_hybrid" ? rclcpp::QoS(1).durability_volatile() : rclcpp::QoS(1).transient_local(),
        [this](const std_msgs::msg::Int32::SharedPtr msg) {
          rt_leg_lift_command_ptr_.writeFromNonRT(msg);
        });
  }

  // Initialize the command subscriber
  cmd_vel_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        rt_cmd_vel_ptr_.writeFromNonRT(msg);
      });

  cmd_pose_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Pose>(
      "/cmd_pose", rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::Pose::SharedPtr msg) {
        rt_cmd_pose_ptr_.writeFromNonRT(msg);
      });

  emergency_stop_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "/emergency_stop", rclcpp::SystemDefaultsQoS(),
      [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
        estop_active_ = true;
        RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      });

  // emergency_stop_reset_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
  //     "/emergency_stop_reset", rclcpp::SystemDefaultsQoS(),
  //     [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  //       if (estop_active_) {
  //         estop_active_ = false;
  //         on_activate(rclcpp_lifecycle::State());
  //         RCLCPP_INFO(get_node()->get_logger(), "Emergency stop released");
  //       }
  //     });

  // Initialize the publishers
  policy_output_publisher_ =
      get_node()->create_publisher<ActionMsg>("~/policy_output", rclcpp::SystemDefaultsQoS());
  rt_policy_output_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ActionMsg>>(policy_output_publisher_);

  position_command_publisher_ =
      get_node()->create_publisher<ActionMsg>("~/position_command", rclcpp::SystemDefaultsQoS());
  rt_position_command_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ActionMsg>>(position_command_publisher_);

  observation_publisher_ =
      get_node()->create_publisher<ObservationMsg>("~/observation", rclcpp::SystemDefaultsQoS());
  rt_observation_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ObservationMsg>>(observation_publisher_);

  // Create IMU latency publishers
  imu_latency_publisher_ = get_node()->create_publisher<std_msgs::msg::Float32>(
      "~/imu_latency_seconds", rclcpp::SystemDefaultsQoS());
  rt_imu_latency_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float32>>(
          imu_latency_publisher_);

  // Create policy inference latency publishers
  policy_inference_latency_publisher_ = get_node()->create_publisher<std_msgs::msg::Float32>(
      "~/policy_inference_latency_seconds", rclcpp::SystemDefaultsQoS());
  rt_policy_inference_latency_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float32>>(
          policy_inference_latency_publisher_);

  RCLCPP_INFO(get_node()->get_logger(), "activate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_error(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::FAILURE;
}

controller_interface::CallbackReturn NeuralController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  rt_cmd_vel_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>>(nullptr);
  rt_cmd_pose_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Pose>>(nullptr);
  rt_leg_lift_command_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::Int32>>(nullptr);
  leg_lift_command_subscriber_ = nullptr;

  for (auto &command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  for (int i = 0; i < kActionSize; i++) {
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.estop_kd);
  }

  // Clear command and state interfaces maps
  command_interfaces_map_.clear();
  state_interfaces_map_.clear();

  // Release underlying command and state interfaces
  command_interfaces_.clear();
  state_interfaces_.clear();

  // Release command and state interfaces from superclass
  release_interfaces();

  RCLCPP_INFO(get_node()->get_logger(), "Deactivate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

void NeuralController::integrate_alignment_motion(double dt) {
  motion_.integrate(hybrid_,dt);
  for(int a=0;a<8;++a) {
    const int row=WheelAlignHybrid::position_rows[a];
    action_[row]=motion_.applied[a];
    command_interfaces_map_.at(params_.joint_names[row]).at("position").get().set_value(motion_.applied[a]);
  }
}

controller_interface::return_type NeuralController::update(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period) {
  // Stop immediately, including during the move to home and between policy ticks.
  // Previously both early returns bypassed the emergency-stop check.
  if (estop_active_) {
    for (auto &command_interface : command_interfaces_) command_interface.set_value(0.0);
    for (int i = 0; i < kActionSize; ++i) {
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd").get().set_value(params_.estop_kd);
    }
    return controller_interface::return_type::OK;
  }
  // When started, return to the default joint positions
  double time_since_init = (time - init_time_).seconds();
  if (time_since_init < params_.init_duration) {
    if (behavior_ == "wheel_align_hybrid") {
      for (int i = 0; i < kActionSize; ++i) {
        hybrid_q_[i] = state_interfaces_map_.at(params_.joint_names[i]).at("position").get().get_value();
        hybrid_qd_[i] = state_interfaces_map_.at(params_.joint_names[i]).at("velocity").get().get_value();
        if (!std::isfinite(hybrid_q_[i]) || !std::isfinite(hybrid_qd_[i]))
          return controller_interface::return_type::ERROR;
      }
    }
    for (int i = 0; i < kActionSize; i++) {
      if (behavior_ == "wheel_align_hybrid" && i % 3 == 2) {
        command_interfaces_map_.at(params_.joint_names[i]).at("velocity").get().set_value(
            WheelAlignHybrid::wheel_pd(hybrid_.hold[i / 3], hybrid_q_[i], hybrid_qd_[i]));
        command_interfaces_map_.at(params_.joint_names[i]).at("kp").get().set_value(0.0);
        command_interfaces_map_.at(params_.joint_names[i]).at("kd").get().set_value(
            params_.kds[i] * params_.gain_multiplier);
        continue;
      }
      // Interpolate between the initial joint positions and the default joint
      // positions
      const double u=std::clamp(time_since_init / params_.init_duration,0.,1.);
      const bool align=behavior_ == "wheel_align_hybrid";
      const double blend=align ? WheelAlignMotion::smooth(u) : u;
      double interpolated_joint_pos = init_joint_pos_.at(i)*(1-blend)+params_.default_joint_pos.at(i)*blend;
      if(align && i%3!=2) {
        const int a=(i/3)*2+i%3;
        hybrid_.applied_position[a]=interpolated_joint_pos;
        motion_.applied[a]=motion_.start[a]=motion_.reference[a]=motion_.desired[a]=interpolated_joint_pos;
        motion_.velocity[a]=(params_.default_joint_pos[i]-init_joint_pos_[i])*30*u*u*(1-u)*(1-u)/params_.init_duration;
      }
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("position")
          .get()
          .set_value(interpolated_joint_pos);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kp")
          .get()
          .set_value(params_.init_kps.at(i));
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd")
          .get()
          .set_value(params_.init_kds.at(i));
    }
    return controller_interface::return_type::OK;
  }

  // After the init_duration has passed, fade in the policy actions
  double time_since_fade_in = (time - init_time_).seconds() - params_.init_duration;
  float fade_in_multiplier = params_.fade_in_duration > 0.0 ?
      std::min(time_since_fade_in / params_.fade_in_duration, 1.0) : 1.0;
  // Hybrid action execution is versioned; no additional policy fade is applied.
  if (behavior_ == "wheel_align_hybrid") fade_in_multiplier = 1.0;
  if (behavior_ == "wheel_align_hybrid") hybrid_elapsed_ += period.seconds();

  // Only get a new action from the policy when repeat_action_counter_ is 0
  repeat_action_counter_ += 1;
  repeat_action_counter_ %= params_.repeat_action;
  if (repeat_action_counter_ != 0) {
    if (behavior_ == "wheel_align_hybrid" && hybrid_.motion_version == 2)
      integrate_alignment_motion(period.seconds());
    return controller_interface::return_type::OK;
  }

  // Get the latest commanded velocities
  auto cmd_vel = rt_cmd_vel_ptr_.readFromRT();
  if (cmd_vel && cmd_vel->get()) {
    const auto bounded = policy_contract_.command(
        {cmd_vel->get()->linear.x, cmd_vel->get()->linear.y, cmd_vel->get()->angular.z});
    cmd_x_vel_ = bounded[0];
    cmd_y_vel_ = bounded[1];
    cmd_yaw_vel_ = bounded[2];
  }

  // Get the latest leg-select command for either one-hot behavior.
  if (behavior_ == "leg_lift" || behavior_ == "wheel_align_hybrid") {
    auto leg_lift_command = rt_leg_lift_command_ptr_.readFromRT();
    if (leg_lift_command && leg_lift_command->get()) {
      int idx = leg_lift_command->get()->data;
      if (idx < 0 || idx >= num_commands_) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "%s command index %d out of range [0, %d)", behavior_.c_str(), idx, num_commands_);
        return controller_interface::return_type::ERROR;
      }
      command_index_ = idx;
    }
  }

  // Get the latest commanded pose
  auto cmd_pose = rt_cmd_pose_ptr_.readFromRT();
  if (policy_contract_.fixed_orientation) {
    const auto &z = policy_contract_.orientation;
    desired_world_z_in_body_frame_ = tf2::Vector3(z[0], z[1], z[2]);
  } else if (cmd_pose && cmd_pose->get()) {
    const auto &pose_msg = *cmd_pose->get();
    tf2::Quaternion q(pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z,
                      pose_msg.orientation.w);
    desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);
    desired_world_z_in_body_frame_ = tf2::quatRotate(q.inverse(), desired_world_z_in_body_frame_);
  }

  // Get the latest observation
  double ang_vel_x = 0;
  double ang_vel_y = 0;
  double ang_vel_z = 0;
  double orientation_w = 0;
  double orientation_x = 0;
  double orientation_y = 0;
  double orientation_z = 0;
  double time_since_measurement_seconds = 0;
  try {
    // read IMU states from hardware interface
    RCLCPP_DEBUG(get_node()->get_logger(), "Attempting to read IMU angular_velocity.x from %s", params_.imu_sensor_name.c_str());
    ang_vel_x = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.x")
                    .get()
                    .get_value();
    ang_vel_y = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.y")
                    .get()
                    .get_value();
    ang_vel_z = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.z")
                    .get()
                    .get_value();
    orientation_w =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.w").get().get_value();
    orientation_x =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.x").get().get_value();
    orientation_y =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.y").get().get_value();
    orientation_z =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.z").get().get_value();

    // Try to read time_since_measurement_seconds if available (optional for simulation)
    auto imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
    if (imu_interfaces.find("time_since_measurement_seconds") != imu_interfaces.end()) {
      time_since_measurement_seconds = imu_interfaces.at("time_since_measurement_seconds").get().get_value();
    } else {
      // Default to 0 if not available (simulation case)
      time_since_measurement_seconds = 0.0;
      RCLCPP_DEBUG_ONCE(get_node()->get_logger(), "time_since_measurement_seconds interface not available, using default value 0.0");
    }

    // Check that the orientation is identity if we are not using the IMU. Use approximate checks
    // to avoid floating point errors
    if (!params_.use_imu) {
      if (std::abs(orientation_w - 1.0) > 1e-3 || std::abs(orientation_x) > 1e-3 ||
          std::abs(orientation_y) > 1e-3 || std::abs(orientation_z) > 1e-3) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "use_imu is false but IMU orientation is not identity");
        return controller_interface::return_type::ERROR;
      }
    } else {
      // Check that the orientation is not identity if we are using the IMU
      if (std::abs(orientation_w - 1.0) < 1e-6 && std::abs(orientation_x) < 1e-6 &&
          std::abs(orientation_y) < 1e-6 && std::abs(orientation_z) < 1e-6) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "use_imu is true but IMU orientation is near identity");
      }
    }

    if (behavior_ == "wheel_align_hybrid" &&
        (!std::isfinite(ang_vel_x) || !std::isfinite(ang_vel_y) || !std::isfinite(ang_vel_z) ||
         !std::isfinite(orientation_x) || !std::isfinite(orientation_y) ||
         !std::isfinite(orientation_z) || !std::isfinite(orientation_w) ||
         orientation_x * orientation_x + orientation_y * orientation_y +
             orientation_z * orientation_z + orientation_w * orientation_w < 1e-12))
      return controller_interface::return_type::ERROR;

    // Calculate the projected gravity vector
    tf2::Quaternion q(orientation_x, orientation_y, orientation_z, orientation_w);
    tf2::Matrix3x3 m(q);
    tf2::Vector3 world_gravity_vector(0, 0, -1);
    tf2::Vector3 projected_gravity_vector = m.inverse() * world_gravity_vector;

    // If the maximum body angle is exceeded, trigger an emergency stop
    if (-projected_gravity_vector[2] < cos(params_.max_body_angle)) {
      estop_active_ = true;
      RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      return controller_interface::return_type::OK;
    }

    // Fill the observation vector
    // Angular velocity
    observation_.at(0) = (float)ang_vel_x;
    observation_.at(1) = (float)ang_vel_y;
    observation_.at(2) = (float)ang_vel_z;
    // Projected gravity vector
    observation_.at(3) = (float)projected_gravity_vector[0];
    observation_.at(4) = (float)projected_gravity_vector[1];
    observation_.at(5) = (float)projected_gravity_vector[2];

    if (behavior_ == "wheel_align_hybrid") {
      for (int i = 0; i < kActionSize; ++i) {
        hybrid_q_[i] = state_interfaces_map_.at(params_.joint_names[i]).at("position").get().get_value();
        hybrid_qd_[i] = state_interfaces_map_.at(params_.joint_names[i]).at("velocity").get().get_value();
        if (!std::isfinite(hybrid_q_[i]) || !std::isfinite(hybrid_qd_[i]))
          return controller_interface::return_type::ERROR;
      }
      const auto previous_phase = hybrid_.phase;
      hybrid_.motion_lower_finished = hybrid_.motion_version == 1 || motion_.lower_finished(hybrid_);
      hybrid_.finish_step(hybrid_q_, hybrid_qd_);
      hybrid_.select_command(command_index_, hybrid_q_);
      if(hybrid_.motion_version==2) motion_.prepare(hybrid_, hybrid_first_step_ ? WheelAlignMotion::control_dt : hybrid_elapsed_);
      if (hybrid_.phase != previous_phase) {
        static constexpr const char *phases[]{"IDLE", "LIFT", "ROTATE", "VERIFY", "LOWER", "HOLD"};
        RCLCPP_INFO(get_node()->get_logger(), "Hybrid %s -> %s; active=%s requested=%s",
            phases[previous_phase], phases[hybrid_.phase],
            command_states_[hybrid_.active_command].c_str(), command_states_[command_index_].c_str());
      }
      for (int i = 0; i < 5; ++i)
        observation_[6 + i] = i == hybrid_.effective_command() ? 1.0f : 0.0f;
      for (int i = 0; i < kActionSize; ++i) {
        observation_[11 + i] = i % 3 == 2 ? WheelAlignHybrid::wrap(hybrid_q_[i]) :
            hybrid_q_[i] - params_.default_joint_pos[i];
        observation_[23 + i] = 0.1 * hybrid_qd_[i];
      }
      for (int k = 0; k < 4; ++k) {
        const double err = WheelAlignHybrid::wrap(hybrid_.target[k] - hybrid_q_[3 * k + 2]);
        observation_[43 + k] = std::sin(err);
        observation_[47 + k] = std::cos(err);
      }
      if(hybrid_.motion_version==2) {
        for(int p=0;p<6;++p) observation_[51+p]=p==hybrid_.phase ? 1.f : 0.f;
        observation_[57]=motion_.progress;
        for(int a=0;a<8;++a) {
          observation_[58+a]=motion_.reference[a]; observation_[66+a]=motion_.applied[a];
          observation_[74+a]=motion_.velocity[a];
        }
      }
    } else if (behavior_ == "leg_lift") {
      // Which-leg-is-up one-hot, driven by joy_util_node's O-button state machine
      for (int i = 0; i < num_commands_; i++) {
        observation_.at(6 + i) = (i == command_index_) ? 1.0f : 0.0f;
      }
    } else {
      // Velocity commands (locomotion and wheel)
      observation_.at(6) = (float)cmd_x_vel_;
      // Lateral velocity: ZERO for the wheel behavior, whatever /cmd_vel carries.
      // These are fixed, non-steerable wheels -- a skid-steer cannot strafe -- so the
      // wheeled policy was trained with vy pinned to 0 and has NEVER seen a nonzero
      // value in this slot. The joystick still publishes linear.y (teleop_twist_joy
      // maps an axis to it with scale 0.5 for the legged policies), and feeding that
      // through put the policy out of distribution, which is what made it behave
      // erratically on hardware when the stick was pushed sideways. Locomotion is
      // unaffected and still uses the real command.
      observation_.at(7) = (behavior_ == "wheel") ? 0.0f : (float)cmd_y_vel_;
      observation_.at(8) = (float)cmd_yaw_vel_;
      if (behavior_ != "wheel") {
        // Orientation commands -- locomotion only; the wheel layout has no such block
        // and slots 9..11 there are already the first joints.
        observation_.at(9) = (float)desired_world_z_in_body_frame_.getX();
        observation_.at(10) = (float)desired_world_z_in_body_frame_.getY();
        observation_.at(11) = (float)desired_world_z_in_body_frame_.getZ();
      }
    }

    // Joint positions
    for (int i = 0; i < kActionSize; i++) {
      // Only include the joint position in the observation if the action type
      // is position
      if (params_.action_types.at(i) == "position") {
        RCLCPP_DEBUG(get_node()->get_logger(), "Attempting to read joint position for %s (index %d)", params_.joint_names.at(i).c_str(), i);
        float joint_pos =
            state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
        observation_.at(joint_position_idx_ + i) = joint_pos - params_.default_joint_pos.at(i);
      }
    }

    // Wheel joints: the loop above skips them (they are velocity-type), which would leave
    // their slots permanently 0. The wheeled policy was trained on normalized, sign-corrected
    // wheel VELOCITY in those slots -- a free-spinning wheel's angle is unbounded and wraps,
    // so position is not a usable input here.
    if (behavior_ == "wheel") {
      for (size_t k = 0; k < wheel_joint_rows_.size(); k++) {
        const int i = wheel_joint_rows_.at(k);
        const float joint_vel =
            state_interfaces_map_.at(params_.joint_names.at(i)).at("velocity").get().get_value();
        observation_.at(joint_position_idx_ + i) =
            static_cast<float>(joint_vel / wheel_velocity_normalizer_ * wheel_forward_sign_.at(k));
      }
    }
  } catch (const std::out_of_range &e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to read states from hardware interface - std::out_of_range exception: %s", e.what());
    
    // Check which interfaces are missing
    RCLCPP_ERROR(get_node()->get_logger(), "=== Debug Information ===");
    
    // Check IMU interface
    if (state_interfaces_map_.find(params_.imu_sensor_name) == state_interfaces_map_.end()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Missing IMU sensor interface: %s", params_.imu_sensor_name.c_str());
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "IMU sensor interface '%s' found", params_.imu_sensor_name.c_str());
      auto &imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
      std::vector<std::string> required_imu = {"angular_velocity.x", "angular_velocity.y", "angular_velocity.z", 
                                               "orientation.x", "orientation.y", "orientation.z", "orientation.w"};
      for (const auto &iface : required_imu) {
        if (imu_interfaces.find(iface) == imu_interfaces.end()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Missing IMU interface: %s.%s", params_.imu_sensor_name.c_str(), iface.c_str());
        }
      }
    }
    
    // Check joint interfaces
    for (size_t i = 0; i < params_.joint_names.size(); i++) {
      const auto &joint_name = params_.joint_names.at(i);
      if (state_interfaces_map_.find(joint_name) == state_interfaces_map_.end()) {
        RCLCPP_ERROR(get_node()->get_logger(), "Missing joint interface: %s", joint_name.c_str());
      } else if (params_.action_types.at(i) == "position") {
        auto &joint_interfaces = state_interfaces_map_.at(joint_name);
        if (joint_interfaces.find("position") == joint_interfaces.end()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Missing position interface for joint: %s", joint_name.c_str());
        }
      }
    }
    
    // List all available interfaces for debugging
    RCLCPP_ERROR(get_node()->get_logger(), "Available state interfaces:");
    for (const auto &[name, interfaces] : state_interfaces_map_) {
      std::string interface_list;
      for (const auto &[iface_name, iface_ref] : interfaces) {
        if (!interface_list.empty()) interface_list += ", ";
        interface_list += iface_name;
      }
      RCLCPP_ERROR(get_node()->get_logger(), "  %s: [%s]", name.c_str(), interface_list.c_str());
    }
    RCLCPP_ERROR(get_node()->get_logger(), "========================");
    
    return controller_interface::return_type::ERROR;
  }

  // Hybrid observe() has no observation clipping; normalization is folded into JSON.
  if (behavior_ == "wheel_align_hybrid" &&
      !std::all_of(observation_.begin(), observation_.end(), [](float x) { return std::isfinite(x); }))
    return controller_interface::return_type::ERROR;
  for (auto &obs : observation_) {
    if (behavior_ == "wheel_align_hybrid") continue;
    obs = std::clamp(obs, static_cast<float>(-params_.observation_limit),
                     static_cast<float>(params_.observation_limit));
  }

  // Check observation for NaNs
  if (contains_nan(observation_)) {
    RCLCPP_ERROR(get_node()->get_logger(), "observation_ contains NaN");
    return controller_interface::return_type::ERROR;
  }

  // Training resets tile the first measured frame through all history slots.
  // Do this once per activation, after sensing, rather than retain stale actions.
  if (seed_history_) {
    seed_observation_history(observation_, single_observation_size_);
    seed_history_ = false;
  }

  // Publish the observation
  if (rt_observation_publisher_->trylock()) {
    // TODO make a custom msg type with header
    // rt_observation_publisher_->msg_.header.stamp = time;
    rt_observation_publisher_->msg_.data = observation_;
    rt_observation_publisher_->unlockAndPublish();
  }

  // Measure the time before policy inference
  auto start_time = std::chrono::high_resolution_clock::now();

  // Perform policy inference
  model_->forward(observation_.data());

  // Measure the time after policy inference
  auto end_time = std::chrono::high_resolution_clock::now();
  auto inference_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  RCLCPP_INFO(get_node()->get_logger(),
              "Policy inference took %.2f ms\tIMU measurement age: %.3f ms",
              inference_duration_us / 1000.0, time_since_measurement_seconds * 1000.0);

  // Shift the observation history to the right by single_observation_size_ for the next control
  // step https://en.cppreference.com/w/cpp/algorithm/rotate
  std::rotate(observation_.rbegin(), observation_.rbegin() + single_observation_size_,
              observation_.rend());

  // Process the actions
  const float *policy_output = model_->getOutputs();

  // Publish the policy output
  if (rt_policy_output_publisher_->trylock()) {
    rt_policy_output_publisher_->msg_.data.resize(policy_action_size_, 0.0);
    for (int i = 0; i < policy_action_size_; i++) {
      rt_policy_output_publisher_->msg_.data.at(i) = policy_output[i];
    }
    // rt_policy_output_publisher_->msg_.header.stamp = time;
    rt_policy_output_publisher_->unlockAndPublish();
  }

  std::array<double, 4> hybrid_wheels{};
  const double hybrid_control_dt = hybrid_first_step_ ? .02 : hybrid_elapsed_;
  if (behavior_ == "wheel_align_hybrid") {
    for (int a = 0; a < 8; ++a) {
      if (!std::isfinite(policy_output[a])) return controller_interface::return_type::ERROR;
      observation_[last_action_idx_ + a] = policy_output[a];
    }
    const bool gate = hybrid_.motion_version==2 ? motion_.ready(hybrid_,hybrid_q_,
        {ang_vel_x,ang_vel_y,ang_vel_z},{observation_[3],observation_[4],observation_[5]}) :
        hybrid_.lift_ready(hybrid_q_, params_.default_joint_pos[3*hybrid_.leg()],
          {ang_vel_x,ang_vel_y,ang_vel_z},observation_[5]);
    hybrid_.begin_step(hybrid_q_, gate, hybrid_first_step_ ? (hybrid_.motion_version==2 ? WheelAlignMotion::control_dt : .02) : hybrid_elapsed_);
    hybrid_first_step_ = false;
    hybrid_elapsed_ = 0.0;
    hybrid_wheels = hybrid_.wheel_commands(hybrid_q_, hybrid_qd_);
  }

  std::array<double,8> hybrid_positions{};
  if (behavior_ == "wheel_align_hybrid") {
    for (int a=0; a<8; ++a) {
      const int row=WheelAlignHybrid::position_rows[a];
      hybrid_positions[a]=std::clamp(double(params_.default_joint_pos[row] + policy_output[a]*params_.action_scales[row]),
          double(params_.joint_lower_limits[row]), double(params_.joint_upper_limits[row]));
    }
    if(hybrid_.motion_version==2) {
      std::array<double,8> raw{}; std::copy_n(policy_output,8,raw.begin());
      motion_.targets(hybrid_,raw);
      motion_.integrate(hybrid_,period.seconds());
      hybrid_positions=motion_.applied;
    } else hybrid_positions=hybrid_.limit_positions(hybrid_positions, hybrid_control_dt);
  }
  for (int i = 0; i < kActionSize; i++) {
    const bool hybrid = behavior_ == "wheel_align_hybrid";
    const bool hybrid_wheel = hybrid && i % 3 == 2;
    const int policy_row = hybrid ? (i / 3) * 2 + i % 3 : i;
    float action = hybrid_wheel ? 0.0f : policy_output[policy_row];
    float action_scale = params_.action_scales.at(i);
    float default_joint_pos = params_.default_joint_pos.at(i);
    float lower_limit = params_.joint_lower_limits.at(i);
    float upper_limit = params_.joint_upper_limits.at(i);

    // Copy policy_output to the observation vector
    if (!hybrid) observation_.at(last_action_idx_ + i) = fade_in_multiplier * action;
    // Scale and de-normalize to get the action vector
    if (params_.action_types.at(i) == "position") {
      float unclipped = fade_in_multiplier * action * action_scale + default_joint_pos;
      action_.at(i) = hybrid ? hybrid_positions[policy_row] : std::clamp(unclipped, lower_limit, upper_limit);
    } else {
      action_.at(i) = hybrid_wheel ? hybrid_wheels[i / 3] : fade_in_multiplier * action * action_scale;
    }

    if (std::isnan(action_.at(i))) {
      RCLCPP_ERROR(get_node()->get_logger(), "action_[%d] is NaN", i);
      return controller_interface::return_type::ERROR;
    }

    // Send the action to the hardware interface
    // Multiply by the gain multiplier to scale the gains to account for real2sim gap
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at(params_.action_types.at(i))
        .get()
        .set_value((double)action_.at(i));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kp")
        .get()
        .set_value(params_.kps.at(i) * params_.gain_multiplier);
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.kds.at(i) * params_.gain_multiplier);
  }

  // Publish the scaled and final position command
  if (rt_position_command_publisher_->trylock()) {
    rt_position_command_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; i++) {
      rt_position_command_publisher_->msg_.data.at(i) = action_.at(i);
    }
    // rt_position_command_publisher_->msg_.header.stamp = time;
    rt_position_command_publisher_->unlockAndPublish();
  }

  // Publish imu latency
  if (rt_imu_latency_publisher_->trylock()) {
    rt_imu_latency_publisher_->msg_.data = time_since_measurement_seconds;
    rt_imu_latency_publisher_->unlockAndPublish();
  }

  if (rt_policy_inference_latency_publisher_->trylock()) {
    rt_policy_inference_latency_publisher_->msg_.data = inference_duration_us / 1000000.0;
    rt_policy_inference_latency_publisher_->unlockAndPublish();
  }

  // Get the policy inference time
  // double policy_inference_time = (get_node()->now() - time).seconds();
  // RCLCPP_INFO(get_node()->get_logger(), "policy inference time: %f",
  // policy_inference_time);

  return controller_interface::return_type::OK;
}

}  // namespace neural_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(neural_controller::NeuralController,
                       controller_interface::ControllerInterface)
