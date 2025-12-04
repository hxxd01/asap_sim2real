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
    void OnEnterRLState() override;  // FSM进入RL状态时的回调
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

    // 轨迹采集相关
    struct TrajectoryData {
        std::vector<std::vector<float>> root_trans_offset;  // [T, 3]
        std::vector<std::vector<std::vector<float>>> pose_aa;  // [T, 35, 3]
        std::vector<std::vector<float>> dof;                // [T, 27]
        std::vector<std::vector<float>> root_rot;           // [T, 4] quaternion xyzw
        std::vector<std::vector<float>> action;             // [T, 27]
        std::vector<bool> terminate;                        // [T]
        std::vector<std::vector<float>> root_lin_vel;       // [T, 3]
        std::vector<std::vector<float>> root_ang_vel;       // [T, 3]
        std::vector<std::vector<float>> dof_vel;            // [T, 27]
        std::vector<float> motion_times;                    // [T]
        float fps = 50.0f;
        
        void clear() {
            root_trans_offset.clear();
            pose_aa.clear();
            dof.clear();
            root_rot.clear();
            action.clear();
            terminate.clear();
            root_lin_vel.clear();
            root_ang_vel.clear();
            dof_vel.clear();
            motion_times.clear();
        }
    };
    
    TrajectoryData traj_data;
    std::vector<TrajectoryData> all_episodes;
    bool is_collecting_trajectory = false;
    bool enable_auto_collection = false;  // 是否启用自动采集
    int trajectory_counter = 0;
    int episode_step_counter = 0;
    int total_step_counter = 0;
    int max_episode_steps = 500;
    int max_total_steps = 10000;
    
    // 早停配置
    double termination_gravity_threshold = 0.85;
    double termination_min_height = 0.35;
    double termination_torque_factor = 3.0;
    
    // 关节轴信息（用于计算pose_aa）
    std::vector<Eigen::Vector3d> dof_axis;  // [27, 3] 每个关节的旋转轴
    void LoadDofAxisFromModel();
    
    // 读取轨迹采集配置
    void LoadTrajectoryConfig(std::string robot_path);
    
    // 采集相关方法
    void InitTrajectoryCollection();
    void CollectTrajectoryData();
    bool CheckTermination(std::string& reason);
    void SaveTrajectoryToPickle(const std::string& filename);
    void ResetEpisode();
    void StartTrajectoryCollection();
    void StopTrajectoryCollection();

};

