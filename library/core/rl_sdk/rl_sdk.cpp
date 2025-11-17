/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"

void RL::StateController(const RobotState<double>* state, RobotCommand<double>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();
}

// TODO
void RL::InitObservations()
{
    this->obs.lin_vel.setZero();  
    
    this->obs.est_lin_vel.setZero();
    this->obs.network_est_lin_vel.setZero();
    
    this->obs.ang_vel.setZero();
    this->obs.gravity_vec << 0.0, 0.0, -1.0;    
    this->obs.base_quat = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);  

    this->obs.dof_pos.resize(this->params.num_of_policy_dofs);
    this->obs.dof_pos = this->params.default_dof_pos;

    this->obs.dof_vel.resize(this->params.num_of_policy_dofs);
    this->obs.dof_vel.setZero();   

    this->obs.actions.resize(this->params.action_dim);
    this->obs.actions.setZero();
    this->obs.observations.resize(this->params.num_observations);
    std::fill(this->obs.observations.begin(), this->obs.observations.end(), 0.0f);

    this->obs_history_buffer.resize(this->params.num_one_step_observations * (this->params.num_history + 1));  
    this->obs_history_buffer.setZero();
}

void RL::InitOutputs()
{
    this->output_dof_tau.resize(this->params.num_of_policy_dofs);
    this->output_dof_tau.setZero();
    this->output_dof_pos = this->params.default_dof_pos;
    this->output_dof_vel.resize(this->params.num_of_policy_dofs);
    this->output_dof_vel.setZero();
}

void RL::InitControl()
{
    this->control.x = 0.0;
    this->control.y = 0.0;
    this->control.yaw = 0.0;

    this->control.current_keyboard == Input::Keyboard::P;
    this->control.current_gamepad == Input::Gamepad::LB_X;
}

void RL::InitRL(std::string robot_path)
{
    this->ReadYamlRL(robot_path);
    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/model/" + this->params.model_name;

    std::cout << "[Loading] model : " << this->params.model_name << " from path : "  << model_path << std::endl;

    // create env
    this->onnx_tensor.onnxEnvPrt.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));

    // create session
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetInterOpNumThreads(1);
    this->onnx_tensor.sessionPtr = std::make_unique<Ort::Session>(*this->onnx_tensor.onnxEnvPrt, model_path.c_str(), sessionOptions);
    // get input and output info
    this->onnx_tensor.inputNames.clear();
    this->onnx_tensor.outputNames.clear();
    this->onnx_tensor.inputShapes.clear();
    this->onnx_tensor.outputShapes.clear();
    Ort::AllocatorWithDefaultOptions allocator;
    std::cout << "count: " << this->onnx_tensor.sessionPtr->GetInputCount() << std::endl;
    for (int i = 0; i < this->onnx_tensor.sessionPtr->GetInputCount(); i++) {
        auto inputnamePtr = this->onnx_tensor.sessionPtr->GetInputNameAllocated(i, allocator);
        this->onnx_tensor.inputNodeNameAllocatedStrings.push_back(std::move(inputnamePtr));
        this->onnx_tensor.inputNames.push_back(this->onnx_tensor.inputNodeNameAllocatedStrings.back().get());
        // inputNames_.push_back(sessionPtr_->GetInputNameAllocated(i,
        // allocator).get());
        this->onnx_tensor.inputShapes.push_back(this->onnx_tensor.sessionPtr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        std::vector<int64_t> shape = this->onnx_tensor.sessionPtr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cerr << "Shape: [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j != shape.size() - 1) { std::cerr << ", "; }
        }
        std::cout << "]" << std::endl;
    }
    for (int i = 0; i < this->onnx_tensor.sessionPtr->GetOutputCount(); i++) {
        auto outputnamePtr = this->onnx_tensor.sessionPtr->GetOutputNameAllocated(i, allocator);
        this->onnx_tensor.outputNodeNameAllocatedStrings.push_back(std::move(outputnamePtr));
        this->onnx_tensor.outputNames.push_back(this->onnx_tensor.outputNodeNameAllocatedStrings.back().get());
        std::cout << this->onnx_tensor.sessionPtr->GetOutputNameAllocated(i, allocator).get() << std::endl;
        this->onnx_tensor.outputShapes.push_back(this->onnx_tensor.sessionPtr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        std::vector<int64_t> shape = this->onnx_tensor.sessionPtr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cerr << "Shape: [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j != shape.size() - 1) { std::cerr << ", "; }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "[Loading] model successfully !!! "  << std::endl;

    this->InitObservations();
    this->InitOutputs();
    this->InitControl();
}

void RL::ComputeOutput()
{

    vector_t actions_scaled = this->obs.actions.array() * this->params.action_scale.array();
    output_dof_pos = actions_scaled + this->params.default_dof_pos.head(this->params.action_dim);

    output_dof_vel = 0.0 * this->obs.dof_vel;  // No velocity control in this case

    output_dof_tau = this->params.rl_kp.array() * (output_dof_pos - this->obs.dof_pos.head(this->params.action_dim)).array()  - this->params.rl_kd.array()  * this->obs.dof_vel.head(this->params.action_dim).array()  ;
}



void RL::TorqueProtect(vector_t origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<double> out_of_range_values;
    for (int i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        double torque_value = origin_output_dof_tau[i];
        double limit_lower = -this->params.torque_limits[i];
        double limit_upper = this->params.torque_limits[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {   
            std::cout << "torque_value : " << torque_value <<  " < limit_lower : " <<  limit_lower <<" = " << (torque_value < limit_lower) << std::endl;
            std::cout << "torque_value : " << torque_value <<  " > limit_upper : " <<  limit_lower <<" = " << (torque_value > limit_upper) << std::endl;

            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (int i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            double value = out_of_range_values[i];
            double limit_lower = -this->params.torque_limits[i];
            double limit_upper = this->params.torque_limits[i];

            std::cout << LOGGER::WARNING << "Torque(" << index << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

// void RL::AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold)
// {
//     float rad2deg = 57.2958;
//     float w, x, y, z;

//     w = quaternion[0];
//     x = quaternion[1];
//     y = quaternion[2];
//     z = quaternion[3];

//     // Calculate roll (rotation around the X-axis)
//     float sinr_cosp = 2 * (w * x + y * z);
//     float cosr_cosp = 1 - 2 * (x * x + y * y);
//     float roll = std::atan2(sinr_cosp, cosr_cosp) * rad2deg;

//     // Calculate pitch (rotation around the Y-axis)
//     float sinp = 2 * (w * y - z * x);
//     float pitch;
//     if (std::fabs(sinp) >= 1)
//     {
//         pitch = std::copysign(90.0, sinp); // Clamp to avoid out-of-range values
//     }
//     else
//     {
//         pitch = std::asin(sinp) * rad2deg;
//     }

//     if (std::fabs(roll) > roll_threshold)
//     {
//         // this->control.SetKeyboard(Input::Keyboard::P);
//         std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
//     }
//     if (std::fabs(pitch) > pitch_threshold)
//     {
//         // this->control.SetKeyboard(Input::Keyboard::P);
//         std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
//     }
// }

#include <termios.h>
#include <sys/ioctl.h>
static bool kbhit()
{
    termios term;
    tcgetattr(0, &term);

    termios term2 = term;
    term2.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &term2);

    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);

    tcsetattr(0, TCSANOW, &term);

    return byteswaiting > 0;
}

void RL::KeyboardInterface()
{
    if (kbhit())
    {
        int c = fgetc(stdin);
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27: this->control.SetKeyboard(Input::Keyboard::Escape); break;
        case 0xE0:
        {
            int ext = fgetc(stdin);
            switch (ext)
            {
            case 72: this->control.SetKeyboard(Input::Keyboard::Up); break;
            case 80: this->control.SetKeyboard(Input::Keyboard::Down); break;
            case 75: this->control.SetKeyboard(Input::Keyboard::Left); break;
            case 77: this->control.SetKeyboard(Input::Keyboard::Right); break;
            default:  break;
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYamlBase(std::string robot_path)
{
    // The config file is located at "rl_sar/src/rl_sar/policy/<robot_path>/base.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/base.yaml";
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[robot_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    this->params.dt = config["dt"].as<double>();
    this->params.decimation = config["decimation"].as<int>();
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();

    auto load_eigen_vector = [this](const YAML::Node& node, const std::string& name) {
        std::vector<double> vec = ReadVectorFromYaml<double>(node);
        if (vec.size() != this->params.num_of_dofs) {
            throw std::runtime_error(name + " size must be " + 
                                std::to_string(this->params.num_of_dofs));
        }
        return Eigen::Map<vector_t>(vec.data(), vec.size()).eval();
    };
    this->params.fixed_kp = load_eigen_vector(config["fixed_kp"], "fixed_kp");
    this->params.fixed_kd = load_eigen_vector(config["fixed_kd"], "fixed_kd");
    this->params.torque_limits = load_eigen_vector(config["torque_limits"], "torque_limits");
    this->params.default_dof_pos = load_eigen_vector(config["default_dof_pos"], "default_dof_pos");

    this->params.joint_names = ReadVectorFromYaml<std::string>(config["joint_names"]);
    this->params.joint_mapping = ReadVectorFromYaml<int>(config["joint_mapping"]);

    std::cout << this->params << std::endl;
}

void RL::ReadYamlRL(std::string robot_path)
{
    // The config file is located at "rl_sar/src/rl_sar/policy/<robot_path>/config.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/config/config.yaml";
    std::cout << "[Loading] config from " << config_path << std::endl;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[robot_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }
    this->params.model_name = config["model_name"].as<std::string>();
    this->params.action_dim = config["action_dim"].as<int>();
    this->params.num_history = config["num_history"].as<int>();
    this->params.num_one_step_observations = config["num_one_step_observations"].as<int>();
    this->params.num_observations = config["num_observations"].as<int>();
    this->params.num_of_policy_dofs = config["num_of_policy_dofs"].as<int>();

    this->params.clip_obs = config["clip_obs"].as<double>();
    this->params.clip_actions = config["clip_actions"].as<double>();

    auto load_eigen_vector = [this](const YAML::Node& node, const std::string& name) {
        if (!node || !node.IsSequence()) { 
            throw std::runtime_error("Missing or invalid YAML node: " + name);
        }
        std::vector<double> vec = ReadVectorFromYaml<double>(node);

        return Eigen::Map<vector_t>(vec.data(), vec.size()).eval();
    };

    this->params.action_scale = load_eigen_vector(config["action_scale"], "action_scale");
    this->params.rl_kp = load_eigen_vector(config["rl_kp"], "rl_kp");
    this->params.rl_kd = load_eigen_vector(config["rl_kd"], "rl_kd");

    this->params.default_dof_pos = load_eigen_vector(config["default_dof_pos"], "default_dof_pos");
    this->params.torque_limits = load_eigen_vector(config["torque_limits"], "torque_limits");

    
    this->params.lin_vel_scale = config["lin_vel_scale"].as<double>();
    this->params.ang_vel_scale = config["ang_vel_scale"].as<double>();
    this->params.dof_pos_scale = config["dof_pos_scale"].as<double>();
    this->params.dof_vel_scale = config["dof_vel_scale"].as<double>();
    if (config["motion_time"]) {
        params.motion_time = config["motion_time"].as<double>();
    } else {
        params.motion_time = 0.0;
    }
    // 打印结果以验证
    std::cout << "####### motion_time: " << params.motion_time << std::endl;
    this->params.commands_scale = load_eigen_vector(config["commands_scale"], "commands_scale");

    this->params.joint_mapping = ReadVectorFromYaml<int>(config["joint_mapping"]);
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    std::string timestamp = ss.str();
    csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}


void RL::CSVLogger(vector_t torque, vector_t tau_est, vector_t joint_pos, vector_t joint_pos_target, vector_t joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << joint_pos[i]<< ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.num_of_policy_dofs; ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

