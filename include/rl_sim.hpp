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

#include <chrono>
#include <thread>
#include <csignal>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>


#include <mujoco/mujoco.h>
#include "joystick.h"
#include "mujoco_utils.cpp"

class RL_Sim : public RL
{
public:
    RL_Sim(std::string robot_name);
    ~RL_Sim();

private:
    // rl
    int motiontime = 0;

    void Forward() override;
    void ComputeObservation() override;

    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command, const RobotState<double> *state) override;
    void SetDefaultPosition() override;  // 设置机器人到默认位置
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;

    mjData *mj_data;
    mjModel *mj_model;
    std::unique_ptr<mj::Simulate> sim;
    Joystick *sys_js;
    int sys_js_max_value = (1 << 15); // 16 bits joystick
    void SetupSysJoystick(std::string device, int bits);
    void GetSysJoystick();

};

