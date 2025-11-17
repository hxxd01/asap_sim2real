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
            this->mj_data->qpos[2] = 0.95;
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
        this->mj_data->qpos[2] = 0.95;
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
        this->mj_data->qpos[2] = 0.95;
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
                this->mj_data->qpos[2] = 0.95;
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
                this->mj_data->qpos[2] = 0.95;
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