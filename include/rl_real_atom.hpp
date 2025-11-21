/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "loop.hpp"
#include "fsm.hpp"
#include "../policy/atom/asap/Asap.hpp"  // Include AsapModel for motion_phase access

#include <csignal>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>

// atom_sdk
#include <atom/common/bridge.h>
#include <atom/rpc/algs_rpc_server.h>

#if KALMAN_FILTER
    #include "kalman_estimator/LinearKalmanFilter.h"
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class RL_Real : public RL
{
public:
    RL_Real();
    ~RL_Real();
    static std::atomic<bool> exit_flag;  // 原子变量，线程安全
private:
    // rl functions
    void Forward() override;
    void ComputeObservation() override;

    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command, const RobotState<double> *state) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    std::string est_csv_filename;
    void EstimationCSVInit(std::string robot_path);
    
    #if KALMAN_FILTER
        legged::KalmanFilterEstimate state_estimator;
        vector_t est_robot_state;
    #endif
    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // others
    int motiontime = 0;
    // std::vector<double> mapped_joint_positions;
    // std::vector<double> mapped_joint_velocities;

    // unitree interface
    void InitLowCmd();

    // Initialize bridge and command structures
    Atom::Bridge bridge;
    Atom::LegCommand leg_command;
    Atom::ArmCommand arm_command;

    // ATOM SDK
    Atom::AlgsRpcServer rpc_server;
    bool isTeleoperation = false;
    void HandleArmSwitch();
    double arm_init_time = 10.0;
    double arm_init_duration = 10.0;
    void InitializeArmPosition();

    // Data logging
    bool log_data = false;
    bool prev_rl_init_done = false;
    bool auto_log_pending = false;

    // std::chrono::steady_clock::time_point last_loop_end;  // 记录上次循环结束时间
    // bool first_loop = true;                               // 首次循环标志

};
