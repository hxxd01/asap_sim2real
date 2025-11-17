/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "fsm_core.hpp"
#include "rl_sdk.hpp"
#include "homie_gym/homie.hpp"
#include "homie_vel_gym/homie_est_vel.hpp"
#include "humanoid_gym/humanoid.hpp"
#include "asap/Asap.hpp"

namespace atom_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        rl.running_percent = 0.0f;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.num_of_dofs; ++i)
        {
            fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 2;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    void Enter() override
    {
        rl.running_percent = 0.0f;
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        if (rl.running_percent < 1.0f)
        {
            rl.running_percent += 1.0f / 500.0f;
            rl.running_percent = std::min(rl.running_percent, 1.0f);

            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                fsm_command->motor_command.q[i] = (1 - rl.running_percent) * rl.now_state.motor_state.q[i] + rl.running_percent * rl.params.default_dof_pos[i];
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = rl.params.fixed_kp[i];
                fsm_command->motor_command.kd[i] = rl.params.fixed_kd[i];
                fsm_command->motor_command.tau[i] = 0;
            }
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.running_percent == 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
            {
                return "RLFSMStatePassive";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStateRL_Locomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
            {
                return "RLFSMStateRL_Vel_Locomotion";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    void Enter() override
    {
        rl.running_percent = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        if (rl.running_percent < 1.0f)
        {
            rl.running_percent += 1.0f / 500.0f;
            rl.running_percent = std::min(rl.running_percent, 1.0f);

            for (int i = 0; i < rl.params.num_of_dofs; ++i)
            {
                fsm_command->motor_command.q[i] = (1 - rl.running_percent) * rl.now_state.motor_state.q[i] + rl.running_percent * rl.start_state.motor_state.q[i];
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = rl.params.fixed_kp[i];
                fsm_command->motor_command.kd[i] = rl.params.fixed_kd[i];
                fsm_command->motor_command.tau[i] = 0;
            }
            std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "Getting down "<< std::fixed << std::setprecision(2) << rl.running_percent * 100.0f << "%" << std::flush;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.running_percent == 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateRL_Locomotion : public RLFSMState
{
public:
    RLFSMStateRL_Locomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRL_Locomotion") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "homie_gym";
        std::string robot_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_path);
            rl.active_model = std::make_unique<HomieModel>();
            rl.active_model_name = rl.active_model->name();
            rl.active_model->reset();  // 重置 counter_step 和 motion_phase
            std::cout << "\n✅ [Enter RL] Reset counter_step & motion_phase to 0" << std::endl;
            
            // ✅ 先设置机器人到默认位置（需要 params 已加载）
            rl.SetDefaultPosition();
            
            // ✅ 位置设置完成后再启用 RL 控制
            rl.rl_init_done = true;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.control.current_keyboard = Input::Keyboard::Num0;
        }
    }

    void Run() override
    {
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller x:" << rl.obs.root_pos[0] << " y:" 
                    << rl.obs.root_pos[1] << " yaw:" << rl.control.yaw  << " height:" << rl.obs.root_pos[2] << " frequency:" << rl.control.frequency << std::flush;

        vector_t _output_dof_pos, _output_dof_vel;
        if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
        {
            for (int i = 0; i < rl.params.action_dim; ++i)
            {
                if (i < _output_dof_pos.size()) 
                {
                    fsm_command->motor_command.q[i] = _output_dof_pos[i];
                }

                if (i < _output_dof_vel.size()) 
                {
                    fsm_command->motor_command.dq[i] = _output_dof_vel[i];
                }
                fsm_command->motor_command.kp[i] = rl.params.rl_kp[i];
                fsm_command->motor_command.kd[i] = rl.params.rl_kd[i];
                fsm_command->motor_command.tau[i] = 0;
            }
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRL_Locomotion";
        }
        // else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        // {
        //     return "RLFSMStateRL_Humanoid_Locomotion";
        // }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateRL_Vel_Locomotion";
        }
        return state_name_;
    }
};


class RLFSMStateRL_Vel_Locomotion : public RLFSMState
{
public:
    RLFSMStateRL_Vel_Locomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRL_Vel_Locomotion") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        // read params from yaml
        rl.config_name = "asap";
        std::string robot_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_path);
            rl.active_model = std::make_unique<AsapModel>();
            rl.active_model_name = rl.active_model->name();
            rl.active_model->reset();  // 重置 counter_step 和 motion_phase
            std::cout << "\n✅ [Enter RL] Reset counter_step & motion_phase to 0" << std::endl;
            
            // ✅ 先设置机器人到默认位置（需要 params 已加载）
            rl.SetDefaultPosition();
            
            // ✅ 位置设置完成后再启用 RL 控制
            rl.rl_init_done = true;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.control.current_keyboard = Input::Keyboard::Num0;
        }
    }

    void Run() override
    {
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller x:" << rl.obs.root_pos[0] << " y:" 
                    << rl.obs.root_pos[1] << " yaw:" << rl.control.yaw  << " height:" << rl.obs.root_pos[2] << " frequency:" << rl.control.frequency << std::flush;

        vector_t _output_dof_pos, _output_dof_vel;
        if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
        {
            for (int i = 0; i < rl.params.action_dim; ++i)
            {
                if (i < _output_dof_pos.size()) 
                {
                    fsm_command->motor_command.q[i] = _output_dof_pos[i];
                }

                if (i < _output_dof_vel.size()) 
                {
                    fsm_command->motor_command.dq[i] = _output_dof_vel[i];
                }
                fsm_command->motor_command.kp[i] = rl.params.rl_kp[i];
                fsm_command->motor_command.kd[i] = rl.params.rl_kd[i];
                fsm_command->motor_command.tau[i] = 0;
            }
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRL_Locomotion";
        }
        // else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        // {
        //     return "RLFSMStateRL_Humanoid_Locomotion";
        // }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateRL_Vel_Locomotion";
        }
        return state_name_;
    }
};



// class RLFSMStateRL_Humanoid_Locomotion : public RLFSMState
// {
// public:
//     RLFSMStateRL_Humanoid_Locomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRL_Humanoid_Locomotion") {}

//     void Enter() override
//     {
//         rl.episode_length_buf = 0;

//         // read params from yaml
//         rl.config_name = "humanoid_gym";
//         std::string robot_path = rl.robot_name + "/" + rl.config_name;
//         try
//         {
//             rl.InitRL(robot_path);
//             rl.active_model = std::make_unique<HumanoidModel>();
//             rl.active_model_name = rl.active_model->name();
//             rl.rl_init_done = true;
//         }
//         catch (const std::exception& e)
//         {
//             std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
//             rl.rl_init_done = false;
//             rl.control.current_keyboard = Input::Keyboard::Num0;
//         }

//         // pos init
//     }

//     void Run() override
//     {
//         std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller x:" << rl.control.x << " y:" 
//                     << rl.control.y << " yaw:" << rl.control.yaw << std::flush;

//         vector_t _output_dof_pos, _output_dof_vel;
//         if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
//         {
//             for (int i = 0; i < rl.params.action_dim; ++i)
//             {
//                 if (i < _output_dof_pos.size()) 
//                 {
//                     fsm_command->motor_command.q[i] = _output_dof_pos[i];
//                 }

//                 if (i < _output_dof_vel.size()) 
//                 {
//                     fsm_command->motor_command.dq[i] = _output_dof_vel[i];
//                 }
//                 fsm_command->motor_command.kp[i] = rl.params.rl_kp[i];
//                 fsm_command->motor_command.kd[i] = rl.params.rl_kd[i];
//                 fsm_command->motor_command.tau[i] = 0;
//             }
//         }
//     }

//     void Exit() override
//     {
//         rl.rl_init_done = false;
//     }

//     std::string CheckChange() override
//     {
//         if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
//         {
//             return "RLFSMStatePassive";
//         }
//         else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
//         {
//             return "RLFSMStateGetDown";
//         }
//         else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
//         {
//             return "RLFSMStateGetUp";
//         }
//         else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
//         {
//             return "RLFSMStateRL_Locomotion";
//         }
//         else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
//         {
//             return "RLFSMStateRL_Humanoid_Locomotion";
//         }
//         return state_name_;
//     }
// };

} // namespace atom_fsm

class AtomFSMFactory : public FSMFactory
{
public:
    AtomFSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<atom_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<atom_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<atom_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRL_Locomotion")
            return std::make_shared<atom_fsm::RLFSMStateRL_Locomotion>(rl);
        // else if (state_name == "RLFSMStateRL_Humanoid_Locomotion")
        //     return std::make_shared<atom_fsm::RLFSMStateRL_Humanoid_Locomotion>(rl);
        else if (state_name == "RLFSMStateRL_Vel_Locomotion")
            return std::make_shared<atom_fsm::RLFSMStateRL_Vel_Locomotion>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "atom"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRL_Locomotion",
            "RLFSMStateRL_Vel_Locomotion",
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(AtomFSMFactory, "RLFSMStatePassive")

