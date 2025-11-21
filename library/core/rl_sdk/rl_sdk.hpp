/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef E1191066_3C61_4377_97EB_712295BB678A
#define E1191066_3C61_4377_97EB_712295BB678A

#ifndef RL_SDK_HPP
#define RL_SDK_HPP

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <string>
#include <exception>
#include <unistd.h>
#include <algorithm>
#include <tbb/concurrent_queue.h>

#include <yaml-cpp/yaml.h>
#include "fsm_core.hpp"
#include "math_rl.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <fstream>   

namespace LOGGER
{
    const char *const INFO    = "\033[0;37m[INFO]\033[0m ";
    const char *const WARNING = "\033[0;33m[WARNING]\033[0m ";
    const char *const ERROR   = "\033[0;31m[ERROR]\033[0m ";
    const char *const DEBUG   = "\033[0;32m[DEBUG]\033[0m ";
    const char *const PARAMS   = "\033[0;32m[PARAMS]\033[0m ";

}

template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<int> mode = std::vector<int>(32, 0);
        std::vector<T> q = std::vector<T>(32, 0.0);
        std::vector<T> dq = std::vector<T>(32, 0.0);
        std::vector<T> tau = std::vector<T>(32, 0.0);
        std::vector<T> kp = std::vector<T>(32, 0.0);
        std::vector<T> kd = std::vector<T>(32, 0.0);
    } motor_command;
};

template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z
        std::vector<T> gyroscope = {0.0, 0.0, 0.0};
        std::vector<T> accelerometer = {0.0, 0.0, 0.0};
    } imu;

    struct MotorState
    {
        std::vector<T> q = std::vector<T>(32, 0.0);
        std::vector<T> dq = std::vector<T>(32, 0.0);
        std::vector<T> ddq = std::vector<T>(32, 0.0);
        std::vector<T> tau_est = std::vector<T>(32, 0.0);
        std::vector<T> cur = std::vector<T>(32, 0.0);
    } motor_state;
};

namespace Input
{
    // Recommend: Num0-GetUp Num9-GetDown N-ToggleNavMode
    //            R-SimReset Enter-SimToggle
    //            M-MotorEnable K-MotorDisable P-MotorPassive
    //            Num1-BaseLocomotion Num2-Num8-Skills(7)
    //            WS-AxisX AD-AxisY QE-AxisYaw Space-AxisClear
    enum class Keyboard
    {
        None = 0,
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Space, Enter, Escape,
        Up, Down, Left, Right
    };

    // Recommend: A-GetUp B-GetDown X-ToggleNavMode Y-None
    //            RB_Y-SimReset RB_X-SimToggle
    //            LB_A-MotorEnable LB_B-MotorDisable LB_X-MotorPassive
    //            RB_DPadUp-BaseLocomotion RB_DPadOthers/LB_DPadOthers-Skills(7)
    //            LY-AxisX LX-AxisY RX-AxisYaw
    enum class Gamepad
    {
        None = 0,
        A, B, X, Y, LB, RB, LStick, RStick, DPadUp, DPadDown, DPadLeft, DPadRight,
        LB_A, LB_B, LB_X, LB_Y, LB_LStick, LB_RStick, LB_DPadUp, LB_DPadDown, LB_DPadLeft, LB_DPadRight,
        RB_A, RB_B, RB_X, RB_Y, RB_LStick, RB_RStick, RB_DPadUp, RB_DPadDown, RB_DPadLeft, RB_DPadRight,
        LB_RB
    };
}

struct Control
{
    Input::Keyboard current_keyboard, last_keyboard;
    Input::Gamepad current_gamepad, last_gamepad;

    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double height = 0.94;
    double frequency = 1.0;

    void SetKeyboard(Input::Keyboard keyboad)
    {
        if (current_keyboard != keyboad)
        {
            last_keyboard = current_keyboard;
            current_keyboard = keyboad;
        }
    }

    void SetGamepad(Input::Gamepad gamepad)
    {
        if (current_gamepad != gamepad)
        {
            last_gamepad = current_gamepad;
            current_gamepad = gamepad;
        }
    }
};

struct ModelParams
{
    std::string model_name;
    double dt;
    int decimation;
    int num_observations;
    int num_one_step_observations;
    int num_history;
    double damping;
    double stiffness;
    vector_t action_scale;
    int num_of_dofs;
    int num_of_policy_dofs;
    int action_dim;
    double lin_vel_scale;
    double ang_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    double clip_obs;
    double clip_actions;
    vector_t torque_limits;
    vector_t rl_kd;
    vector_t rl_kp;
    vector_t fixed_kp;
    vector_t fixed_kd;
    vector_t commands_scale;
    vector_t default_dof_pos;
    std::vector<std::string> joint_names;
    std::vector<int> joint_mapping;

    double motion_time;
};


inline std::ostream& operator<<(std::ostream& os, const ModelParams& params) {
    os << LOGGER::PARAMS << "dt: " << params.dt << "\n";
    os << LOGGER::PARAMS << "decimation: " << params.decimation << "\n";
    os << LOGGER::PARAMS << "num_of_dofs: " << params.num_of_dofs << "\n";

    auto print_vector = [&os](const auto& vec, const std::string& name) {
        os << LOGGER::PARAMS << name << ":\n";
        for (size_t i = 0; i < vec.size(); ++i) {
            os << "  [" << i << "] = " << vec[i] << "\n";
        }
        os << "\n";
    };

    print_vector(params.fixed_kp, "fixed_kp");
    print_vector(params.fixed_kd, "fixed_kd");
    print_vector(params.torque_limits, "torque_limits");
    print_vector(params.default_dof_pos, "default_dof_pos");

    print_vector(params.joint_names, "joint_names");
    print_vector(params.joint_mapping, "joint_mapping");

    return os;
}



struct Observations
{
    vector3_t root_pos;
    vector3_t lin_vel;
    vector3_t est_lin_vel;
    vector3_t network_est_lin_vel;

    vector3_t ang_vel;
    vector3_t gravity_vec;
    quaternion_t base_quat;

    // vector_t commands;
    vector_t dof_pos;
    vector_t dof_vel;
    vector_t actions; 

    std::vector<tensor_element_t> observations;
};


struct OnnxTensor
{
    std::shared_ptr<Ort::Env> onnxEnvPrt;
    std::unique_ptr<Ort::Session> sessionPtr;
    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
};



// ================== RobotModel 抽象基类 ==================
class RobotModel {
public:
    virtual ~RobotModel() = default;
    virtual void Forward(const ModelParams& params, OnnxTensor& onnx_tensor, Observations& obs) = 0;
    virtual vector_t compute_observation(const ModelParams& params,
                                const RobotState<double>& robot_state, 
                                const Control& control, 
                                Observations& obs)  = 0;
    virtual std::string name() const = 0;
    virtual vector_t get_contact_state(){ return vector_t(0);}
    virtual void reset() {}  // 默认空实现，子类可以重写
};

class RL
{
public:
    RL(){};
    ~RL() {};

    ModelParams params;
    Observations obs;
    vector_t obs_history_buffer;


    RobotState<double> robot_state;
    RobotCommand<double> robot_command;
    tbb::concurrent_queue<vector_t> output_dof_pos_queue;
    tbb::concurrent_queue<vector_t> output_dof_vel_queue;
    tbb::concurrent_queue<vector_t> output_dof_tau_queue;

    FSM fsm;
    RobotState<double> start_state;
    RobotState<double> now_state;
    float running_percent = 0.0f;
    bool rl_init_done = false;

    // init
    virtual void InitObservations();
    virtual void InitOutputs();
    void InitControl();
    void InitRL(std::string robot_path);

    // rl functions
    virtual void Forward() = 0;
    // vector_t ComputeObservation();
    virtual void ComputeObservation() = 0;
    virtual void GetState(RobotState<double> *state) = 0;

    // 状态机：当前活跃模型名
    std::string current_model_name = "idle";

    // 当前 RL 所用 model 实例
    std::unique_ptr<RobotModel> active_model;
    std::string active_model_name;


    virtual void SetCommand(const RobotCommand<double> *command, const RobotState<double> *state) = 0;
    void StateController(const RobotState<double> *state, RobotCommand<double> *command);
    void ComputeOutput();
    // vector3_t QuatRotateInverse(const quaternion_t& q, const vector3_t& v);
    
    // 设置机器人到默认位置（用于进入RL策略时）
    virtual void SetDefaultPosition() {}
    // yaml params
    void ReadYamlBase(std::string robot_name);
    void ReadYamlRL(std::string robot_name);

    // csv logger
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(vector_t joint_pos, vector_t joint_vel, vector_t tau_est,
                   vector_t cmd_q, vector_t cmd_tau, float motion_phase);

    // control
    Control control;
    void KeyboardInterface();

    // history buffer
    vector_t history_obs;

    // others
    std::string robot_name, config_name;
    bool simulation_running = true;
    std::string ang_vel_type = "ang_vel_body";  // "ang_vel_world" or "ang_vel_body"
    unsigned long long episode_length_buf = 0;
    float motion_length = 0.0;

    // protect func
    void TorqueProtect(vector_t origin_output_dof_tau);
    // void AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold);

    // rl module
    OnnxTensor onnx_tensor;

    // output buffer
    vector_t output_dof_tau;
    vector_t output_dof_pos;
    vector_t output_dof_vel;
};

class RLFSMState : public FSMState
{
public:
    RLFSMState(RL& rl, const std::string& name)
        : FSMState(name), rl(rl), fsm_state(nullptr), fsm_command(nullptr) {}
    RL& rl;
    const RobotState<double>* fsm_state;
    RobotCommand<double>* fsm_command;
};

template <typename T>
T clamp(T value, T min, T max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}



#endif // RL_SDK_HPP


#endif /* E1191066_3C61_4377_97EB_712295BB678A */
