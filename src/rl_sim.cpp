/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"
#include <cmath>


RL_Sim::RL_Sim(std::string robot_name){

    this->robot_name = robot_name;
    this->ang_vel_type = "ang_vel_body";

    std::cout << LOGGER::INFO << "Launching mujoco..." << std::endl;

    std::printf("MuJoCo version %s\n", mj_versionString());
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/assets/" + this->robot_name + "_description/atom.xml";

    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());

    while (1)
    {
        if (d)
        {
            std::cout << "Mujoco data is prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16);
    this->ReadYamlBase(this->robot_name);
    
    if (this->mj_model && this->mj_data && this->sim)
    {
        sim->run = 0;
        // Wait for sim->Load() to complete in render thread
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        {
            const std::unique_lock<std::recursive_mutex> lock(sim->mtx);
            // Set target timestep for PhysicsLoop to use
            SetTargetTimestep(this->params.dt);
            
            // Set MuJoCo timestep to match control period dt for 200Hz simulation frequency
            // This ensures: simulation=200Hz, control=200Hz, RL=50Hz (with decimation=4)
            // Note: timestep may be reset by sim->Load() in render thread, PhysicsLoop will fix it
            this->mj_model->opt.timestep = this->params.dt;
            m->opt.timestep = this->params.dt;
            std::cout << LOGGER::INFO << "Set MuJoCo timestep to " << this->mj_model->opt.timestep << " s" << std::endl;
            std::cout << LOGGER::INFO << "Simulation frequency: " << (1.0 / this->mj_model->opt.timestep) << " Hz" << std::endl;
            std::cout << LOGGER::INFO << "Control frequency: " << (1.0 / this->params.dt) << " Hz" << std::endl;
            std::cout << LOGGER::INFO << "RL frequency: " << (1.0 / (this->params.dt * this->params.decimation)) << " Hz" << std::endl;
            std::cout << LOGGER::INFO << "Verified global m->opt.timestep = " << m->opt.timestep << " s" << std::endl;
            
            mj_resetData(this->mj_model, this->mj_data);

            this->mj_data->qpos[0] = 0.0;
            this->mj_data->qpos[1] = 0.0;
            this->mj_data->qpos[2] = 0.976;
            this->mj_data->qpos[3] = 1.0;
            this->mj_data->qpos[4] = 0.0;
            this->mj_data->qpos[5] = 0.0;
            this->mj_data->qpos[6] = 0.0;

            for (int i = 0; i < this->params.num_of_dofs; ++i)
            {
                this->mj_data->qpos[7 + i] = this->params.default_dof_pos[i];
            }
            for (int i = 0; i < this->mj_model->nv; ++i) { this->mj_data->qvel[i] = 0.0; }
            for (int i = 0; i < this->mj_model->nv; ++i) { this->mj_data->qacc[i] = 0.0; }
            for (int i = 0; i < this->mj_model->nu; ++i) { this->mj_data->ctrl[i] = 0.0; }

            mj_forward(this->mj_model, this->mj_data);
        }

        sim->run = 1;
        std::cout << LOGGER::INFO << "Initialized model at timestep=" << this->params.dt << " s" << std::endl;
    }

    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }

    this->InitOutputs();
    this->InitControl();

    // 从MuJoCo模型加载关节轴信息（必须在MuJoCo初始化之后）
    LoadDofAxisFromModel();
    
    // 加载轨迹采集配置
    this->LoadTrajectoryConfig(this->robot_name);
 
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    sim->RenderLoop();
    physicsthreadhandle.join();

}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::ComputeObservation() {

    auto proprioObs = this->active_model->compute_observation(this->params, this->robot_state, this->control, this->obs);
    
    this->obs_history_buffer.tail(this->obs_history_buffer.size() - this->params.num_one_step_observations) =
        this->obs_history_buffer.head(this->obs_history_buffer.size() - this->params.num_one_step_observations);
    this->obs_history_buffer.head(this->params.num_one_step_observations) = proprioObs;

    int idx = 0;
    const int num_history = 4;
    
    for (int i = 0; i < 84; i++) {
        this->obs.observations[idx++] = static_cast<tensor_element_t>(proprioObs[i]);
    }
    
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 0; i < 27; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }
    
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 27; i < 30; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }
    
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 30; i < 57; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }
    
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 57; i < 84; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }
    
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 84; i < 87; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }
    
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + 87]);
    }
    
    for (int i = 84; i < 88; i++) {
        this->obs.observations[idx++] = static_cast<tensor_element_t>(proprioObs[i]);
    }
    
    scalar_t obsMin = -this->params.clip_obs;
    scalar_t obsMax = this->params.clip_obs;
    std::transform(this->obs.observations.begin(), this->obs.observations.end(), this->obs.observations.begin(),
                    [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

void RL_Sim::GetState(RobotState<double> *state)
{
    if (this->mj_data && this->sim)
    {
        this->GetSysJoystick();

        const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
        
        // Ensure timestep is correct (may be reset by model reloading)
        if (this->mj_model && std::abs(this->mj_model->opt.timestep - this->params.dt) > 1e-6) {
            this->mj_model->opt.timestep = this->params.dt;
            m->opt.timestep = this->params.dt;
        }

        this->obs.root_pos[0] = this->mj_data->qpos[0];
        this->obs.root_pos[1] = this->mj_data->qpos[1];
        this->obs.root_pos[2] = this->mj_data->qpos[2];

        this->obs.lin_vel[0] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 13];
        this->obs.lin_vel[1] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 14];
        this->obs.lin_vel[2] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 15];

        state->imu.quaternion[0] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 0];
        state->imu.quaternion[1] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 1];
        state->imu.quaternion[2] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 2];
        state->imu.quaternion[3] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 3];

        state->imu.gyroscope[0] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 4];
        state->imu.gyroscope[1] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 5];
        state->imu.gyroscope[2] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 6];

        state->imu.accelerometer[0] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 7];
        state->imu.accelerometer[1] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 8];
        state->imu.accelerometer[2] = this->mj_data->sensordata[3 * this->params.num_of_dofs + 9];

        for (int i = 0; i < this->params.num_of_dofs; ++i)
        {
            int mujoco_idx = this->params.joint_mapping[i];
            state->motor_state.q[i] = this->mj_data->sensordata[mujoco_idx];
            state->motor_state.dq[i] = this->mj_data->sensordata[mujoco_idx + this->params.num_of_dofs];
            state->motor_state.tau_est[i] = this->mj_data->sensordata[mujoco_idx + 2 * this->params.num_of_dofs];
        }
    }
}

void RL_Sim::SetCommand(const RobotCommand<double> *command, const RobotState<double> *state)
{
    if (!this->mj_data || !this->sim) {
        return;
    }

    const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
    
    // Ensure timestep is correct (may be reset by model reloading)
    if (this->mj_model && std::abs(this->mj_model->opt.timestep - this->params.dt) > 1e-6) {
        this->mj_model->opt.timestep = this->params.dt;
        m->opt.timestep = this->params.dt;
    }

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        int mujoco_idx = this->params.joint_mapping[i];

        double pos_err = command->motor_command.q[i] - this->mj_data->sensordata[mujoco_idx];
        double vel_err = 0.0 - this->mj_data->sensordata[mujoco_idx + this->params.num_of_dofs];
        
        double u = command->motor_command.tau[i] +
                  command->motor_command.kp[i] * pos_err +
                  command->motor_command.kd[i] * vel_err;
   
        double lim = this->params.torque_limits[i];
        if (u >  lim) u =  lim;
        if (u < -lim) u = -lim;
    
        this->mj_data->ctrl[mujoco_idx] = u;
    }
}

void RL_Sim::SetDefaultPosition()
{
    if (this->mj_model && this->mj_data && this->sim)
    {
        const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
        
        this->mj_data->qpos[0] = 0.0;
        this->mj_data->qpos[1] = 0.0;
        this->mj_data->qpos[2] = 0.976;
        this->mj_data->qpos[3] = 1.0;
        this->mj_data->qpos[4] = 0.0;
        this->mj_data->qpos[5] = 0.0;
        this->mj_data->qpos[6] = 0.0;
   
        for (int i = 0; i < this->params.num_of_dofs; ++i)
        {
            this->mj_data->qpos[7 + i] = this->params.default_dof_pos[i];
        }
      
        for (int i = 0; i < this->mj_model->nv; ++i)
        {
            this->mj_data->qvel[i] = 0.0;
        }

        for (int i = 0; i < this->mj_model->nv; ++i)
        {
            this->mj_data->qacc[i] = 0.0;
        }
        
        for (int i = 0; i < this->mj_model->nu; ++i)
        {
            this->mj_data->ctrl[i] = 0.0;
        }
        
        mj_step(this->mj_model, this->mj_data);

        this->mj_data->qpos[0] = 0.0;
        this->mj_data->qpos[1] = 0.0;
        this->mj_data->qpos[2] = 0.976;
        this->mj_data->qpos[3] = 1.0;
        this->mj_data->qpos[4] = 0.0;
        this->mj_data->qpos[5] = 0.0;
        this->mj_data->qpos[6] = 0.0;
        for (int i = 0; i < this->params.num_of_dofs; ++i)
        {
            this->mj_data->qpos[7 + i] = this->params.default_dof_pos[i];
        }
        
        for (int i = 0; i < this->mj_model->nv; ++i)
        {
            this->mj_data->qvel[i] = 0.0;
        }
  
        for (int i = 0; i < this->mj_model->nv; ++i)
        {
            this->mj_data->qacc[i] = 0.0;
        }
        
        mj_forward(this->mj_model, this->mj_data);
        
        std::cout << "\n✅ [Enter RL] Set robot to default position" << std::endl;
        std::cout << "   Base: [" << this->mj_data->qpos[0] << ", " << this->mj_data->qpos[1] << ", " << this->mj_data->qpos[2] << "]" << std::endl;
        std::cout << "   Joint[0:3]: [" << this->mj_data->qpos[7] << ", " << this->mj_data->qpos[8] << ", " << this->mj_data->qpos[9] << "]" << std::endl;
    }
}

void RL_Sim::RobotControl()
{
    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data && this->sim)
        {
            bool was_running = simulation_running;
            simulation_running = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            {
                const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
                
                this->mj_data->qpos[0] = 0.0;
                this->mj_data->qpos[1] = 0.0;
                this->mj_data->qpos[2] = 0.976;
                this->mj_data->qpos[3] = 1.0;
                this->mj_data->qpos[4] = 0.0;
                this->mj_data->qpos[5] = 0.0;
                this->mj_data->qpos[6] = 0.0;
                
                for (int i = 0; i < this->params.num_of_dofs; ++i)
                {
                    this->mj_data->qpos[7 + i] = this->params.default_dof_pos[i];
                }
                
                for (int i = 0; i < this->mj_model->nv; ++i)
                {
                    this->mj_data->qvel[i] = 0.0;
                }
                
                for (int i = 0; i < this->mj_model->nu; ++i)
                {
                    this->mj_data->ctrl[i] = 0.0;
                }
                
                mj_step(this->mj_model, this->mj_data);
                
                this->mj_data->qpos[0] = 0.0;
                this->mj_data->qpos[1] = 0.0;
                this->mj_data->qpos[2] = 0.976;
                this->mj_data->qpos[3] = 1.0;
                this->mj_data->qpos[4] = 0.0;
                this->mj_data->qpos[5] = 0.0;
                this->mj_data->qpos[6] = 0.0;
                for (int i = 0; i < this->params.num_of_dofs; ++i)
                {
                    this->mj_data->qpos[7 + i] = this->params.default_dof_pos[i];
                }
                
                for (int i = 0; i < this->mj_model->nv; ++i)
                {
                    this->mj_data->qvel[i] = 0.0;
                }
                
                mj_forward(this->mj_model, this->mj_data);
                
                std::cout << "\n🔄 [Reset] Resetting to default position..." << std::endl;
                std::cout << "   Base: [" << this->mj_data->qpos[0] << ", " << this->mj_data->qpos[1] << ", " << this->mj_data->qpos[2] << "]" << std::endl;
                std::cout << "   Quat: [" << this->mj_data->qpos[3] << ", " << this->mj_data->qpos[4] << ", " << this->mj_data->qpos[5] << ", " << this->mj_data->qpos[6] << "]" << std::endl;
                std::cout << "   Joint[0:3]: [" << this->mj_data->qpos[7] << ", " << this->mj_data->qpos[8] << ", " << this->mj_data->qpos[9] << "]" << std::endl;
            }
            std::cout << "   default_dof_pos[0:3]: [" << this->params.default_dof_pos[0] << ", " << this->params.default_dof_pos[1] << ", " << this->params.default_dof_pos[2] << "]" << std::endl;
            this->obs.actions.setZero();
            this->obs_history_buffer.setZero();
            if (this->active_model) {
                std::cout << "   ✅ Resetting active_model (counter_step & motion_phase)" << std::endl;
                this->active_model->reset();
            } else {
                std::cout << "   ⚠️ active_model is nullptr (not in RL state)" << std::endl;
            }
            
            simulation_running = was_running;
            if (was_running) {
                std::cout << "   ▶️  Resuming simulation..." << std::endl;
            }
        }
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (simulation_running)
    {
        this->motiontime++;

        if (this->control.current_keyboard == Input::Keyboard::W)
        {
            this->control.x += 0.1;
            this->control.current_keyboard = this->control.last_keyboard;
        }
        if (this->control.current_keyboard == Input::Keyboard::S)
        {
            this->control.x -= 0.1;
            this->control.current_keyboard = this->control.last_keyboard;
        }
        if (this->control.current_keyboard == Input::Keyboard::A)
        {
            this->control.y += 0.1;
            this->control.current_keyboard = this->control.last_keyboard;
        }
        if (this->control.current_keyboard == Input::Keyboard::D)
        {
            this->control.y -= 0.1;
            this->control.current_keyboard = this->control.last_keyboard;
        }
        if (this->control.current_keyboard == Input::Keyboard::Q)
        {
            this->control.yaw += 0.1;
            this->control.current_keyboard = this->control.last_keyboard;
        }
        if (this->control.current_keyboard == Input::Keyboard::E)
        {
            this->control.yaw -= 0.1;
            this->control.current_keyboard = this->control.last_keyboard;
        }
        if (this->control.current_keyboard == Input::Keyboard::Space)
        {
            this->control.x = 0;
            this->control.y = 0;
            this->control.yaw = 0;
            this->control.current_keyboard = this->control.last_keyboard;
        }
         
         // T 键：开始/停止轨迹采集
         if (this->control.current_keyboard == Input::Keyboard::T)
         {
             if (!is_collecting_trajectory) {
                 StartTrajectoryCollection();
             } else {
                 StopTrajectoryCollection();
             }
             this->control.current_keyboard = this->control.last_keyboard;
         }

        this->GetState(&this->robot_state);
        this->StateController(&this->robot_state, &this->robot_command);
        this->SetCommand(&this->robot_command, &this->robot_state);
    }
}


void RL_Sim::SetupSysJoystick(std::string device, int bits)
{
    this->sys_js = new Joystick(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "System Joystick open failed." << std::endl;
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    this->sys_js->getState();

    if (this->sys_js->button_[0]) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js->button_[1]) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js->button_[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js->button_[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js->button_[4]) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js->button_[5]) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js->button_[9]) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js->button_[10]) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js->axis_[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js->axis_[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js->axis_[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js->axis_[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js->button_[4] && this->sys_js->button_[0]) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js->button_[4] && this->sys_js->button_[1]) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js->button_[4] && this->sys_js->button_[2]) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js->button_[4] && this->sys_js->button_[3]) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js->button_[4] && this->sys_js->button_[9]) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js->button_[4] && this->sys_js->button_[10]) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js->button_[4] && this->sys_js->axis_[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js->button_[4] && this->sys_js->axis_[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js->button_[4] && this->sys_js->axis_[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js->button_[4] && this->sys_js->axis_[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js->button_[5] && this->sys_js->button_[0]) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js->button_[5] && this->sys_js->button_[1]) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js->button_[5] && this->sys_js->button_[2]) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js->button_[5] && this->sys_js->button_[3]) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js->button_[5] && this->sys_js->button_[9]) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js->button_[5] && this->sys_js->button_[10]) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js->button_[5] && this->sys_js->axis_[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js->button_[5] && this->sys_js->axis_[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js->button_[5] && this->sys_js->axis_[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js->button_[5] && this->sys_js->axis_[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js->button_[4] && this->sys_js->button_[5]) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = -double(this->sys_js->axis_[1]) / this->sys_js_max_value * 0.6;
    this->control.y = double(this->sys_js->axis_[0]) / this->sys_js_max_value * 0.6;
    this->control.yaw = double(this->sys_js->axis_[3]) / this->sys_js_max_value * 0.6;
}



void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;

        this->GetState(&this->robot_state);
        
        this->ComputeObservation();
        this->Forward();
        this->ComputeOutput();

        output_dof_pos_queue.push(this->output_dof_pos);
        output_dof_vel_queue.push(this->output_dof_vel);
        output_dof_tau_queue.push(this->output_dof_tau);

         // ========== 轨迹采集逻辑 ==========
         if (is_collecting_trajectory) {
             // 采集当前帧数据
             CollectTrajectoryData();
             
             // 检查早停条件
             std::string reason;
             bool should_terminate = CheckTermination(reason);
             
             // 标记终止
             if (!traj_data.terminate.empty()) {
                 traj_data.terminate.back() = should_terminate;
             }
             
             episode_step_counter++;
             total_step_counter++;
             
             // Episode 结束条件
             bool episode_complete = should_terminate || 
                                    (episode_step_counter >= max_episode_steps);
             bool collection_complete = total_step_counter >= max_total_steps;
             
             if (episode_complete) {
                 if (should_terminate) {
                     std::cout << "🛑 [Termination] " << reason 
                               << " at step " << episode_step_counter << std::endl;
                 }
                 
                 // Reset episode
                 ResetEpisode();
                 
                 // 如果达到总步数，停止采集
                 if (collection_complete) {
                     StopTrajectoryCollection();
                 }
             }
         }
         // ========== 轨迹采集逻辑结束 ==========
 
#ifdef CSV_LOGGER
    vector_t tau_est = Eigen::Map<const vector_t>(
        this->robot_state.motor_state.tau_est.data(),
        this->robot_state.motor_state.tau_est.size()
    );        
    this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

void RL_Sim::Forward()
{
    this->active_model->Forward(this->params, this->onnx_tensor, this->obs);
}
 
 // ========== 轨迹采集功能实现 ==========
 
// 从MuJoCo模型加载关节轴信息
void RL_Sim::LoadDofAxisFromModel() {
    if (!mj_model) {
        std::cerr << "[WARNING] MuJoCo model not loaded, cannot load dof_axis" << std::endl;
        return;
    }
    
    dof_axis.clear();
    
    // 遍历所有关节，提取旋转轴
    for (int i = 0; i < mj_model->njnt; i++) {
        // 跳过自由关节
        int jnt_type = mj_model->jnt_type[i];
        if (jnt_type == mjJNT_FREE || jnt_type == mjJNT_BALL) {
            continue;
        }
        
        // 获取关节轴（存储在 jnt_axis 中，每个关节3个值）
        if (jnt_type == mjJNT_HINGE || jnt_type == mjJNT_SLIDE) {
            Eigen::Vector3d axis;
            axis[0] = mj_model->jnt_axis[3*i + 0];
            axis[1] = mj_model->jnt_axis[3*i + 1];
            axis[2] = mj_model->jnt_axis[3*i + 2];
            dof_axis.push_back(axis);
        }
    }
    
    std::cout << "[Trajectory] Loaded " << dof_axis.size() << " joint axes" << std::endl;
    if (dof_axis.size() > 0 && dof_axis.size() <= 3) {
        for (size_t i = 0; i < dof_axis.size(); i++) {
            std::cout << "  Joint " << i << " axis: [" 
                      << dof_axis[i][0] << ", " 
                      << dof_axis[i][1] << ", " 
                      << dof_axis[i][2] << "]" << std::endl;
        }
    }
}

// 读取轨迹采集配置
void RL_Sim::LoadTrajectoryConfig(std::string robot_path) {
     // 尝试多种可能的配置路径
     std::vector<std::string> possible_paths = {
         std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/asap/config/config.yaml",
         std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/config/config.yaml"
     };
     
     bool config_loaded = false;
     
     for (const auto& config_path : possible_paths) {
         try {
             YAML::Node root = YAML::LoadFile(config_path);
             
             // 尝试多种节点名称: "atom/asap", "atom", "asap"
             YAML::Node config;
             if (root["atom/asap"]) {
                 config = root["atom/asap"];
             } else if (root[robot_path + "/asap"]) {
                 config = root[robot_path + "/asap"];
             } else if (root[robot_path]) {
                 config = root[robot_path];
             } else {
                 continue;  // 尝试下一个路径
             }
             
             // 读取配置项
             if (config["enable_trajectory_collection"]) {
                 enable_auto_collection = config["enable_trajectory_collection"].as<bool>();
             }
             if (config["max_episode_steps"]) {
                 max_episode_steps = config["max_episode_steps"].as<int>();
             }
             if (config["max_total_steps"]) {
                 max_total_steps = config["max_total_steps"].as<int>();
             }
             if (config["termination_gravity_threshold"]) {
                 termination_gravity_threshold = config["termination_gravity_threshold"].as<double>();
             }
             if (config["termination_min_height"]) {
                 termination_min_height = config["termination_min_height"].as<double>();
             }
             if (config["termination_torque_factor"]) {
                 termination_torque_factor = config["termination_torque_factor"].as<double>();
             }
             
             std::cout << LOGGER::INFO << "Trajectory collection config loaded from: " << config_path << std::endl;
             std::cout << "  Auto collection: " << (enable_auto_collection ? "✅ ENABLED" : "DISABLED") << std::endl;
             if (enable_auto_collection) {
                 std::cout << "  🎬 按2进入RL策略时将自动开始采集" << std::endl;
             } else {
                 std::cout << "  按T键手动开始采集" << std::endl;
             }
             std::cout << "  Max episode steps: " << max_episode_steps << std::endl;
             std::cout << "  Max total steps: " << max_total_steps << std::endl;
             std::cout << "  Gravity threshold: " << termination_gravity_threshold << std::endl;
             std::cout << "  Min height: " << termination_min_height << std::endl;
             std::cout << "  Torque factor: " << termination_torque_factor << std::endl;
             
             config_loaded = true;
             break;
             
         } catch (const std::exception& e) {
             continue;  // 尝试下一个路径
         }
     }
     
     if (!config_loaded) {
         std::cout << LOGGER::WARNING << "Failed to load trajectory config, using defaults" << std::endl;
         std::cout << "  按T键手动开始采集" << std::endl;
     }
 }
 
 // 初始化轨迹采集
 void RL_Sim::InitTrajectoryCollection() {
     traj_data.clear();
     episode_step_counter = 0;
     std::cout << "\n📊 [Trajectory] Episode initialized" << std::endl;
 }
 
 // 开始轨迹采集
 void RL_Sim::OnEnterRLState() {
     std::cout << "\n========================================" << std::endl;
     std::cout << "[DEBUG] 🔵 OnEnterRLState called!" << std::endl;
     std::cout << "  enable_auto_collection = " << enable_auto_collection << std::endl;
     std::cout << "  is_collecting_trajectory = " << is_collecting_trajectory << std::endl;
     std::cout << "========================================\n" << std::endl;
     std::flush(std::cout);
     
     // 如果启用自动采集且当前未在采集，则开始采集
     if (enable_auto_collection && !is_collecting_trajectory) {
         std::cout << "[DEBUG] 🟢 Starting trajectory collection..." << std::endl;
         std::flush(std::cout);
         StartTrajectoryCollection();
     } else {
         std::cout << "[DEBUG] 🔴 NOT starting collection (enable=" 
                   << enable_auto_collection << ", already_collecting=" 
                   << is_collecting_trajectory << ")" << std::endl;
         std::flush(std::cout);
     }
 }
 
 void RL_Sim::StartTrajectoryCollection() {
     std::cout << "\n[Trajectory] 🎬 StartTrajectoryCollection() called" << std::endl;
     std::flush(std::cout);
     
     if (is_collecting_trajectory) {
         std::cout << "⚠️  [Trajectory] Already collecting" << std::endl;
         return;
     }
     
     is_collecting_trajectory = true;
     std::cout << "✅ [Trajectory] is_collecting_trajectory set to TRUE" << std::endl;
     std::flush(std::cout);
     total_step_counter = 0;
     all_episodes.clear();
     InitTrajectoryCollection();
     
     std::cout << "\n🎬 [Trajectory] Collection STARTED" << std::endl;
     std::cout << "  Max episode steps: " << max_episode_steps << std::endl;
     std::cout << "  Max total steps: " << max_total_steps << std::endl;
 }
 
 // 停止轨迹采集并保存
 void RL_Sim::StopTrajectoryCollection() {
     if (!is_collecting_trajectory) {
         std::cout << "⚠️  [Trajectory] Not collecting" << std::endl;
         return;
     }
     
     // 保存当前 episode（如果有数据）
     if (!traj_data.motion_times.empty()) {
         all_episodes.push_back(traj_data);
         std::cout << "📦 [Trajectory] Saved final episode: " 
                   << traj_data.motion_times.size() << " frames" << std::endl;
     }
     
     is_collecting_trajectory = false;
     
     // 生成时间戳文件名
     auto now = std::chrono::system_clock::now();
     std::time_t now_c = std::chrono::system_clock::to_time_t(now);
     std::stringstream ss;
     ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
     std::string timestamp = ss.str();
     
     std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + this->robot_name + 
                           "/trajectory_" + timestamp + ".json";
     
     SaveTrajectoryToPickle(filename);
     
     std::cout << "\n🏁 [Trajectory] Collection STOPPED" << std::endl;
     std::cout << "  Total episodes: " << all_episodes.size() << std::endl;
     std::cout << "  Total steps: " << total_step_counter << std::endl;
     std::cout << "\n💡 Convert to pickle:" << std::endl;
     std::cout << "  python scripts/save_trajectory_pickle.py " << filename << std::endl;
 }
 
 // 采集当前帧数据
 void RL_Sim::CollectTrajectoryData() {
     if (!this->mj_data || !this->mj_model) return;
     
     const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
     
     // 获取状态
     double* qpos = this->mj_data->qpos;
     double* qvel = this->mj_data->qvel;
     
     // 1. root_trans_offset [3]
     std::vector<float> root_trans = {
         static_cast<float>(qpos[0]),
         static_cast<float>(qpos[1]),
         static_cast<float>(qpos[2])
     };
     traj_data.root_trans_offset.push_back(root_trans);
     
     // 2. root_rot (quaternion xyzw) [4]
     std::vector<float> root_rot = {
         static_cast<float>(qpos[4]),  // x
         static_cast<float>(qpos[5]),  // y
         static_cast<float>(qpos[6]),  // z
         static_cast<float>(qpos[3])   // w
     };
     traj_data.root_rot.push_back(root_rot);
     
     // 3. dof [27]
     std::vector<float> dof(27);
     for (int i = 0; i < 27; i++) {
         dof[i] = static_cast<float>(qpos[7 + i]);
     }
     traj_data.dof.push_back(dof);
     
     // 4. pose_aa [35, 3] - axis-angle representation
     Eigen::Quaterniond quat(qpos[3], qpos[4], qpos[5], qpos[6]);  // w,x,y,z
     Eigen::AngleAxisd aa(quat);
     Eigen::Vector3d root_rotvec = aa.angle() * aa.axis();
     
     std::vector<std::vector<float>> pose_aa(35, std::vector<float>(3, 0.0f));
     pose_aa[0] = {static_cast<float>(root_rotvec[0]), 
                   static_cast<float>(root_rotvec[1]), 
                   static_cast<float>(root_rotvec[2])};
     
      // Joint axis-angles (27 joints) - 关节角度 * 旋转轴方向
      int num_joints = std::min(27, static_cast<int>(dof_axis.size()));
      for (int i = 0; i < num_joints; i++) {
          double angle = qpos[7 + i];
          Eigen::Vector3d joint_aa = angle * dof_axis[i];  // 角度 * 轴方向
          pose_aa[i + 1] = {static_cast<float>(joint_aa[0]), 
                            static_cast<float>(joint_aa[1]), 
                            static_cast<float>(joint_aa[2])};
      }
      
      // 剩余的关节（如果dof_axis不足27个）保持为0
      for (int i = num_joints; i < 27; i++) {
          pose_aa[i + 1] = {0.0f, 0.0f, 0.0f};
      }
     
     traj_data.pose_aa.push_back(pose_aa);
     
     // 5. action [27]
     std::vector<float> action(27);
     for (int i = 0; i < 27; i++) {
         action[i] = static_cast<float>(this->obs.actions[i]);
     }
     traj_data.action.push_back(action);
     
     // 6. root_lin_vel [3]
     std::vector<float> root_lin_vel = {
         static_cast<float>(qvel[0]),
         static_cast<float>(qvel[1]),
         static_cast<float>(qvel[2])
     };
     traj_data.root_lin_vel.push_back(root_lin_vel);
     
     // 7. root_ang_vel [3]
     std::vector<float> root_ang_vel = {
         static_cast<float>(qvel[3]),
         static_cast<float>(qvel[4]),
         static_cast<float>(qvel[5])
     };
     traj_data.root_ang_vel.push_back(root_ang_vel);
     
     // 8. dof_vel [27]
     std::vector<float> dof_vel(27);
     for (int i = 0; i < 27; i++) {
         dof_vel[i] = static_cast<float>(qvel[6 + i]);
     }
     traj_data.dof_vel.push_back(dof_vel);
     
     // 9. motion_times
     float current_time = trajectory_counter * this->params.dt * this->params.decimation;
     traj_data.motion_times.push_back(current_time);
     
     // 10. terminate
     traj_data.terminate.push_back(false);
     
     trajectory_counter++;
 }
 
 // 检查终止条件
 bool RL_Sim::CheckTermination(std::string& reason) {
     if (!this->mj_data || !this->mj_model) return false;
     
     const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
     
     double* qpos = this->mj_data->qpos;
     
     // 1. 重力方向检查
     Eigen::Quaterniond quat(qpos[3], qpos[4], qpos[5], qpos[6]);  // w,x,y,z
     Eigen::Vector3d gravity_world(0, 0, -1);
     Eigen::Vector3d gravity_base = quat.inverse() * gravity_world;
     
     if (std::abs(gravity_base[0]) > termination_gravity_threshold || 
         std::abs(gravity_base[1]) > termination_gravity_threshold) {
         reason = "gravity violation (gx=" + std::to_string(gravity_base[0]) + 
                  ", gy=" + std::to_string(gravity_base[1]) + ")";
         return true;
     }
     
     // 2. 高度检查
     double base_height = qpos[2];
     if (base_height < termination_min_height) {
         reason = "low height (h=" + std::to_string(base_height) + ")";
         return true;
     }
     
     // 3. 力矩检查
     for (int i = 0; i < 27; i++) {
         double torque = this->mj_data->ctrl[i];
         double limit = this->params.torque_limits[i];
         if (std::abs(torque) > termination_torque_factor * limit) {
             reason = "torque violation (joint" + std::to_string(i) + 
                      ": " + std::to_string(torque) + " > " + 
                      std::to_string(termination_torque_factor * limit) + ")";
             return true;
         }
     }
     
     return false;
 }
 
 // Reset当前episode
 void RL_Sim::ResetEpisode() {
     // 保存当前episode
     if (!traj_data.motion_times.empty()) {
         all_episodes.push_back(traj_data);
         std::cout << "\n📦 [Trajectory] Episode " << all_episodes.size() 
                   << " saved: " << traj_data.motion_times.size() << " frames, "
                   << traj_data.motion_times.back() << "s" << std::endl;
     }
     
     // 重置物理状态
     SetDefaultPosition();
     
     // 重置观测缓冲
     this->obs.actions.setZero();
     this->obs_history_buffer.setZero();
     
     // 重置模型状态
     if (this->active_model) {
         this->active_model->reset();
         std::cout << "  ✅ Model reset (counter_step=0, motion_phase=0)" << std::endl;
     }
     
     // 初始化新episode
     InitTrajectoryCollection();
     trajectory_counter = 0;
 }
 
 // 保存为JSON格式
 void RL_Sim::SaveTrajectoryToPickle(const std::string& filename) {
     std::ofstream file(filename);
     if (!file.is_open()) {
         std::cerr << "❌ Failed to open file: " << filename << std::endl;
         return;
     }
     
     file << std::fixed << std::setprecision(6);
     file << "{\n";
     file << "  \"episode_count\": " << all_episodes.size() << ",\n";
     
     for (size_t ep_idx = 0; ep_idx < all_episodes.size(); ep_idx++) {
         const auto& ep = all_episodes[ep_idx];
         size_t T = ep.motion_times.size();
         
         file << "  \"motion" << ep_idx << "\": {\n";
         
         // root_trans_offset
         file << "    \"root_trans_offset\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[" << ep.root_trans_offset[t][0] << "," 
                  << ep.root_trans_offset[t][1] << "," << ep.root_trans_offset[t][2] << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         // pose_aa (simplified output)
         file << "    \"pose_aa\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[";
             for (size_t j = 0; j < 35; j++) {
                 file << "[" << ep.pose_aa[t][j][0] << "," 
                      << ep.pose_aa[t][j][1] << "," << ep.pose_aa[t][j][2] << "]";
                 if (j < 34) file << ",";
             }
             file << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         // dof, root_rot, action, terminate, velocities (简化格式)
         file << "    \"dof\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[";
             for (size_t i = 0; i < 27; i++) {
                 file << ep.dof[t][i];
                 if (i < 26) file << ",";
             }
             file << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"root_rot\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[" << ep.root_rot[t][0] << "," << ep.root_rot[t][1] << "," 
                  << ep.root_rot[t][2] << "," << ep.root_rot[t][3] << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"action\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[";
             for (size_t i = 0; i < 27; i++) {
                 file << ep.action[t][i];
                 if (i < 26) file << ",";
             }
             file << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"terminate\": [";
         for (size_t t = 0; t < T; t++) {
             file << (ep.terminate[t] ? "true" : "false");
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"root_lin_vel\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[" << ep.root_lin_vel[t][0] << "," 
                  << ep.root_lin_vel[t][1] << "," << ep.root_lin_vel[t][2] << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"root_ang_vel\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[" << ep.root_ang_vel[t][0] << "," 
                  << ep.root_ang_vel[t][1] << "," << ep.root_ang_vel[t][2] << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"dof_vel\": [";
         for (size_t t = 0; t < T; t++) {
             file << "[";
             for (size_t i = 0; i < 27; i++) {
                 file << ep.dof_vel[t][i];
                 if (i < 26) file << ",";
             }
             file << "]";
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"motion_times\": [";
         for (size_t t = 0; t < T; t++) {
             file << ep.motion_times[t];
             if (t < T-1) file << ",";
         }
         file << "],\n";
         
         file << "    \"fps\": " << ep.fps << "\n";
         file << "  }";
         if (ep_idx < all_episodes.size() - 1) file << ",";
         file << "\n";
     }
     
     file << "}\n";
     file.close();
     
     std::cout << "\n✅ [Trajectory] Saved to: " << filename << std::endl;
     std::cout << "📊 Total episodes: " << all_episodes.size() << std::endl;
     
     int total_frames = 0;
     for (const auto& ep : all_episodes) {
         total_frames += ep.motion_times.size();
     }
     std::cout << "📊 Total frames: " << total_frames << std::endl;
     std::cout << "⏱️  Total duration: " << (total_frames * 0.02f) << "s" << std::endl;
 }
 
 // ========== 轨迹采集功能实现结束 ==========


void signalHandler(int signum)
{
    pthread_exit(NULL);
}


int main(int argc, char **argv)
{
    std::string robot_name = "atom";
    signal(SIGINT, signalHandler);

    RL_Sim rl_sar(robot_name);
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}