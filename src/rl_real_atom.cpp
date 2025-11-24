/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_atom.hpp"
std::atomic<bool> RL_Real::exit_flag{false};  // 初始化值为false

RL_Real::RL_Real()
{
    rpc_server.Start(51234);
    rpc_server.Init();
    // SwitchUpperLimbControl callback
    rpc_server.RegisterSetTeleSwitchCallback([&](bool is_on) {
        rpc_server.SetTeleSwitch(is_on);
        return 1;
    });
    rpc_server.Start(51234);

    // read params from yaml
    this->robot_name = "atom";
    this->ReadYamlBase(this->robot_name);
    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name)) {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr) { this->fsm = *fsm_ptr; }
    } else {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }
    // actions_.resize(12);

    // baseMotor_.resize(32, 0.0);
    // init robot
    this->InitLowCmd();
    this->InitOutputs();
    this->InitControl();
#if KALMAN_FILTER
    this->est_robot_state = vector_t::Zero(2*18);
#endif

    // loop

    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
    EstimationCSVInit(this->robot_name);

}

RL_Real::~RL_Real()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}


void RL_Real::EstimationCSVInit(std::string robot_path)
{
    this->est_csv_filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/policy/" + robot_path + "/estimation_real";

    // Uncomment these lines if need timestamp for file name
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    std::string timestamp = ss.str();
    this->est_csv_filename += "_" + timestamp;

    this->est_csv_filename += ".csv";
    std::ofstream file(this->est_csv_filename.c_str());
    // file << "est_lin_x" << ",";
    // file << "est_lin_y" << ",";
    // file << "est_lin_z" << ",";

    // file << "actual_lin_x" << ",";
    // file << "actual_lin_y" << ",";
    // file << "actual_lin_z" << ",";

    // file << "est_contact_force_left" << ",";
    // file << "est_contact_force_right" << ",";

    // file << "desired_contact_left" << ",";
    // file << "desired_contact_right" << ",";
    file << "x" << ",";
    file << "y" << ",";
    file << "z" << ",";

    file << "r" << ",";
    file << "p" << ",";
    file << "yaw" << ",";
    for(int i = 0; i < 12; ++i) { file << "qpos" << i << ","; }

    file << "est_lin_x" << ",";
    file << "est_lin_y" << ",";
    file << "est_lin_z" << ",";

    file << "ang_x" << ",";
    file << "ang_y" << ",";
    file << "ang_z" << ",";

    for(int i = 0; i < 12; ++i) { file << "qvel" << i << ","; }

    file << "network_lin_x" << ",";
    file << "network_lin_y" << ",";
    file << "network_lin_z" << ",";
    file << "est_contact_force_left" << ",";
    file << "est_contact_force_right" << ",";
    file << std::endl;

    file.close();
}


void RL_Real::ComputeObservation()
{
    auto proprioObs = this->active_model->compute_observation(this->params, this->robot_state, this->control, this->obs);
    // 打印关键维度
    std::cout << "[DEBUG] proprioObs.size()=" << proprioObs.size() << std::endl;
    std::cout << "[DEBUG] obs_history_buffer.size()=" << this->obs_history_buffer.size() << std::endl;
    std::cout << "[DEBUG] num_one_step_observations=" << this->params.num_one_step_observations << std::endl;
    std::cout << "[DEBUG] num_history=" << this->params.num_history << std::endl;
    std::cout << "[DEBUG] obs.observations.size()=" << this->obs.observations.size() << std::endl;

    // Update history buffer - move old data back, add new data at head
    this->obs_history_buffer.tail(this->obs_history_buffer.size() - this->params.num_one_step_observations) =
        this->obs_history_buffer.head(this->obs_history_buffer.size() - this->params.num_one_step_observations);
    this->obs_history_buffer.head(this->params.num_one_step_observations) = proprioObs;

    int idx = 0;
    const int num_history = this->params.num_history;  // Use config value instead of hardcoded 4

    // Current frame observations (84 elements)
    for (int i = 0; i < 84; i++) {
        this->obs.observations[idx++] = static_cast<tensor_element_t>(proprioObs[i]);
    }

    // History: DOF positions (27 * 4)
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 0; i < 27; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }

    // History: DOF velocities (3 * 4)
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 27; i < 30; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }

    // History: Gravity and other obs (27 * 4)
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 30; i < 57; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }

    // History: More observations (27 * 4)
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 57; i < 84; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }

    // History: IMU info (3 * 4)
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        for (int i = 84; i < 87; i++) {
            this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + i]);
        }
    }

    // History: Motion phase (1 * 4)
    for (int t = 1; t <= num_history; t++) {
        int frame_offset = t * this->params.num_one_step_observations;
        this->obs.observations[idx++] = static_cast<tensor_element_t>(this->obs_history_buffer[frame_offset + 87]);
    }

    // Current frame additional info (4)
    for (int i = 84; i < 88; i++) {
        this->obs.observations[idx++] = static_cast<tensor_element_t>(proprioObs[i]);
    }

    // Limit observation range
    scalar_t obsMin = -this->params.clip_obs;
    scalar_t obsMax = this->params.clip_obs;
    std::transform(this->obs.observations.begin(), this->obs.observations.end(), this->obs.observations.begin(),
                    [obsMin, obsMax](scalar_t x) { return std::max(obsMin, std::min(obsMax, x)); });
}

void RL_Real::GetState(RobotState<double> *state)
{
    const std::shared_ptr<const Atom::BaseState> base_state_ptr = bridge.GetNewestBaseStatePtr();
    const std::shared_ptr<const Atom::JointState> js_tmp_ptr = bridge.GetNewestJointStatePtr();
    const std::shared_ptr<const RemoteCommand> remote_tmp_ptr = bridge.GetNewestRemoteCommandPtr();
    const std::shared_ptr<const Atom::ArmJointState> ajs_tmp_ptr = bridge.GetNewestArmStatePtr();  // arm

    if (remote_tmp_ptr->button_A_) this->control.SetGamepad(Input::Gamepad::A);          // get up
    if (remote_tmp_ptr->button_B_) this->control.SetGamepad(Input::Gamepad::B);          // get down
    if (remote_tmp_ptr->button_SELECT_) this->control.SetGamepad(Input::Gamepad::LB_X);  // passive 按backup就是进阻尼状态
    if (remote_tmp_ptr->button_L1U_) {
        if (this->control.height < 0.94) {
            this->control.height += 0.001;
        } else {
            this->control.height = 0.94;
        }
    }
    if (remote_tmp_ptr->button_L1D_) {
        if (this->control.height > 0.4) {
            this->control.height -= 0.001;
        } else {
            this->control.height = 0.4;
        }
    }

    if (remote_tmp_ptr->button_START_ && !this->log_data)
    {
        if (!this->rl_init_done || this->params.num_of_policy_dofs <= 0)
        {
            std::cout << std::endl << LOGGER::WARNING
                      << "Cannot enable logging: RL model not initialized yet." << std::endl;
        }
        else
        {
            this->CSVInit(this->robot_name);
            this->log_data = true;
            this->auto_log_pending = false;
            std::cout << std::endl << LOGGER::INFO << "Data logging ENABLED (500Hz) - Press L2 to stop" << std::endl;
        }
    }
    if (remote_tmp_ptr->button_L2_ && this->log_data)
    {
        this->log_data = false;
        std::cout << std::endl << LOGGER::INFO << "Data logging DISABLED" << std::endl;
    }

    if (remote_tmp_ptr->button_R1U_) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);  // r1+up进入模型

    if (remote_tmp_ptr->button_R1L_) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);  // r1+down进入站立

    this->control.x = remote_tmp_ptr->lin_vel[0] * 1.0;
    this->control.y = remote_tmp_ptr->lin_vel[1] * 0.0;
    this->control.yaw = remote_tmp_ptr->yaw_vel * 0.8;

    state->imu.quaternion[0] = base_state_ptr->qua[0];  // w
    state->imu.quaternion[1] = base_state_ptr->qua[1];  // x
    state->imu.quaternion[2] = base_state_ptr->qua[2];  // y
    state->imu.quaternion[3] = base_state_ptr->qua[3];  // z

    for (int i = 0; i < 3; ++i) { state->imu.gyroscope[i] = base_state_ptr->w[i]; }

    // Leg joints (12 joints: indices 0-11)
    for (int i = 0; i < 12; ++i) {
        state->motor_state.q[i] = js_tmp_ptr->q[i];
        state->motor_state.dq[i] = js_tmp_ptr->dq[i];
        state->motor_state.tau_est[i] = js_tmp_ptr->tau[i];
    }

    // Arm joints (17 joints: indices 12-28)
    // Note: Policy controls 15 arm joints (12-26), head joints (27-28) are handled separately
    for (int i = 0; i < 17; ++i) {
        state->motor_state.q[i + 12] = ajs_tmp_ptr->q[i];
        state->motor_state.dq[i + 12] = ajs_tmp_ptr->dq[i];
        state->motor_state.tau_est[i + 12] = ajs_tmp_ptr->tau[i];
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command, const RobotState<double> *state)
{
    // Send leg commands (12 joints: indices 0-11)
    for (int i = 0; i < 12; ++i) {
        // this->leg_command.q_ref[i] = (command->motor_command.q[i] - baseMotor_[i]);
        this->leg_command.q_ref[i] = (command->motor_command.q[i]);

        this->leg_command.dq_ref[i] = command->motor_command.dq[i];
        this->leg_command.kp[i] = command->motor_command.kp[i];
        this->leg_command.kd[i] = command->motor_command.kd[i];
        
        // Clip torque to limits
        double tau = command->motor_command.tau[i];
        double tau_limit = this->params.torque_limits[i];
        if (tau > tau_limit) tau = tau_limit;
        if (tau < -tau_limit) tau = -tau_limit;
        this->leg_command.tau_forward[i] = tau;
    }

    // Send arm commands (15 joints controlled by policy: indices 12-26)
    // Note: Policy controls 27 DOFs total (12 legs + 15 upper body)
    // Real robot has 17 arm joints, but policy only controls 15
    for (int i = 12; i < this->params.num_of_dofs; ++i) {
        // this->arm_command.q_ref[i - 12] = (command->motor_command.q[i] - baseMotor_[i - 1]);
        this->arm_command.q_ref[i - 12] = (command->motor_command.q[i]);
        this->arm_command.dq_ref[i - 12] = command->motor_command.dq[i];
        this->arm_command.kp[i - 12] = command->motor_command.kp[i];
        this->arm_command.kd[i - 12] = command->motor_command.kd[i];
        
        // Clip torque to limits
        double tau = command->motor_command.tau[i];
        double tau_limit = this->params.torque_limits[i];
        if (tau > tau_limit) tau = tau_limit;
        if (tau < -tau_limit) tau = -tau_limit;
        this->arm_command.tau_forward[i - 12] = tau;
    }
    
    // Fill head joints (indices 15-16 in ArmCommand, corresponding to real joints 27-28)
    // Keep head stable at current position with moderate stiffness
    const int head_start_idx = 15;  // ArmCommand[15] = HeadPitch, [16] = HeadRoll
    for (int i = 0; i < 2; ++i) {
        this->arm_command.q_ref[head_start_idx + i] = 0.0;  // Keep current position
        this->arm_command.dq_ref[head_start_idx + i] = 0.0;
        this->arm_command.kp[head_start_idx + i] = 100.0;  // Moderate stiffness
        this->arm_command.kd[head_start_idx + i] = 4.0;
        this->arm_command.tau_forward[head_start_idx + i] = 0.0;
    }
    
    int fsm_id = 2;
    bridge.SetNewestFsmCommand(fsm_id);
    bridge.SetNewestLegCommand(this->leg_command);

    HandleArmSwitch();
    if (!isTeleoperation) {
        if (arm_init_time < arm_init_duration) {
            InitializeArmPosition();
            arm_init_time += this->params.dt;
        }
        bridge.SetNewestArmCommand(this->arm_command);
    } else {
        arm_init_time = 0;
    }
}

void RL_Real::InitializeArmPosition()
{
    double ratio = static_cast<double>(arm_init_time) / arm_init_duration;
    const auto &q_vec = this->robot_state.motor_state.q;
    std::vector<double> q_tmp(q_vec.begin() + 12, q_vec.begin() + 29);
    Eigen::Map<const Eigen::VectorXd> q_current_d(q_tmp.data(), 17);
    Eigen::VectorXf q_current = q_current_d.cast<float>();
    Eigen::Matrix<float, 17, 1> q_ref = Eigen::Matrix<float, 17, 1>::Zero();
    this->arm_command.q_ref = ratio * (q_ref - q_current) + q_current;
}

void RL_Real::RobotControl()
{
    // Reset logic (R key or RB+Y gamepad)
    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        std::cout << "\n🔄 [Reset Request] Resetting observation buffers and model state..." << std::endl;
        
        // Reset observation buffers
        this->obs.actions.setZero();
        this->obs_history_buffer.setZero();
        
        // Reset active model (counter_step and motion_phase)
        if (this->active_model) {
            this->active_model->reset();
            std::cout << "   ✅ Model reset complete (counter_step=0, motion_phase=0)" << std::endl;
        } else {
            std::cout << "   ⚠️  active_model is nullptr (not in RL state)" << std::endl;
        }
        
        std::cout << "   📊 Current robot state:" << std::endl;
        std::cout << "      Quat(w,x,y,z): [" << this->robot_state.imu.quaternion[0] << ", " 
                  << this->robot_state.imu.quaternion[1] << ", " 
                  << this->robot_state.imu.quaternion[2] << ", " 
                  << this->robot_state.imu.quaternion[3] << "]" << std::endl;
        std::cout << "      Joint[0:3]: [" << this->robot_state.motor_state.q[0] << ", " 
                  << this->robot_state.motor_state.q[1] << ", " 
                  << this->robot_state.motor_state.q[2] << "]" << std::endl;
        
        this->control.current_keyboard = this->control.last_keyboard;
    }
    
    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::W) {
        this->control.x += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::S) {
        this->control.x -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::A) {
        this->control.y += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::D) {
        this->control.y -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q) {
        this->control.yaw += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::E) {
        this->control.yaw -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (this->control.current_keyboard == Input::Keyboard::Z) {
        this->control.height -= 0.01;

        if (this->control.height <= 0.4) { this->control.height = 0.4; }
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::X) {
        this->control.height += 0.01;

        if (this->control.height >= 0.98) { this->control.height = 0.98; }
        this->control.current_keyboard = this->control.last_keyboard;
    }

    if (this->control.current_keyboard == Input::Keyboard::Space) {
        this->control.x = 0;
        this->control.y = 0;
        this->control.yaw = 0;
        this->control.height = 0.9;
        this->control.frequency = 1.0;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    bool rl_just_started = this->rl_init_done && !this->prev_rl_init_done;
    bool rl_just_stopped = !this->rl_init_done && this->prev_rl_init_done;
    this->prev_rl_init_done = this->rl_init_done;

    if (rl_just_started)
    {
        this->auto_log_pending = true;
    }
    if (rl_just_stopped)
    {
        this->auto_log_pending = false;
        if (this->log_data)
        {
            this->log_data = false;
            std::cout << std::endl << LOGGER::INFO << "RL exited - Data logging DISABLED" << std::endl;
        }
    }

    // if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    // {
    //     this->control.navigation_mode = !this->control.navigation_mode;
    //     std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    //     this->control.current_keyboard = this->control.last_keyboard;
    // }

    this->GetState(&this->robot_state);
    
    // Log data at 500Hz (every 2 control cycles at 1000Hz)
    if(this->log_data && this->rl_init_done) {

            // Prepare current joint states (27 DOFs)
            const int logged_dofs = this->params.num_of_policy_dofs;
            vector_t joint_pos = vector_t::Zero(logged_dofs);
            vector_t joint_vel = vector_t::Zero(logged_dofs);
            vector_t tau_est = vector_t::Zero(logged_dofs);
            
            for(int i = 0; i < logged_dofs; ++i) {
                joint_pos[i] = this->robot_state.motor_state.q[i];
                joint_vel[i] = this->robot_state.motor_state.dq[i];
                tau_est[i] = this->robot_state.motor_state.tau_est[i];
            }
            
            int control_dim = this->params.action_dim;
            control_dim = std::min(control_dim, static_cast<int>(this->output_dof_pos.size()));
            control_dim = std::min(control_dim, static_cast<int>(this->obs.dof_pos.size()));
            control_dim = std::min(control_dim, static_cast<int>(this->obs.dof_vel.size()));
            control_dim = std::min(control_dim, static_cast<int>(this->params.rl_kp.size()));
            control_dim = std::min(control_dim, static_cast<int>(this->params.rl_kd.size()));
            control_dim = std::max(control_dim, 0);

            vector_t cmd_tau_full = vector_t::Zero(logged_dofs);
            if (control_dim > 0) {
                vector_t tau_cmd = this->params.rl_kp.head(control_dim).array() *
                                   (this->output_dof_pos.segment(0, control_dim) - this->obs.dof_pos.segment(0, control_dim)).array()
                                   - this->params.rl_kd.head(control_dim).array() *
                                     this->obs.dof_vel.segment(0, control_dim).array();
                cmd_tau_full.segment(0, control_dim) = tau_cmd;
            }
            
            // Clip cmd_tau to torque limits before logging (DISABLED - log raw values)
            // for(int i = 0; i < logged_dofs; ++i) {
            //     double tau_limit = this->params.torque_limits[i];
            //     if (cmd_tau_full[i] > tau_limit) cmd_tau_full[i] = tau_limit;
            //     if (cmd_tau_full[i] < -tau_limit) cmd_tau_full[i] = -tau_limit;
            // }
            this->output_dof_tau = cmd_tau_full;

            vector_t cmd_pos_full = vector_t::Zero(logged_dofs);
            int pos_dim = std::min(static_cast<int>(this->output_dof_pos.size()), logged_dofs);
            if (pos_dim > 0) {
                cmd_pos_full.segment(0, pos_dim) = this->output_dof_pos.segment(0, pos_dim);
            }

            // Get motion phase from active model (cast to AsapModel)
            float motion_phase = 0.0f;
            if (this->active_model) {
                auto asap_model = dynamic_cast<AsapModel*>(this->active_model.get());
                if (asap_model) {
                    motion_phase = static_cast<float>(asap_model->get_motion_phase());
                }
            }
            
            this->CSVLogger(joint_pos, joint_vel, tau_est, cmd_pos_full, cmd_tau_full, motion_phase);
        
    }



    this->StateController(&this->robot_state, &this->robot_command);

    // for (size_t i = 0; i < 12; i++)
    // {
    //     std::cout << "after motor_state.q[" << i << "] = " << this->robot_state.motor_state.q[i] << std::endl;
    //     std::cout << "after motor_state.dq[" << i << "] = " << this->robot_state.motor_state.dq[i] << std::endl;
    //     std::cout << "after motor_state.tau_est[" << i << "] = " << this->robot_state.motor_state.tau_est[i] << std::endl;
    // }

    this->SetCommand(&this->robot_command, &this->robot_state);
}

void RL_Real::HandleArmSwitch()
{
    if (!rpc_server.GetTeleSwitch()) {
        isTeleoperation = false;
    } else {
        isTeleoperation = true;
    }
    // std::cout << " rpc_server: " << rpc_server.GetTeleSwitch() << ", initTime: " << arm_init_time << std::endl;
}

void RL_Real::RunModel()
{
    if (this->rl_init_done) {
        // auto loop_start = std::chrono::steady_clock::now();

        this->episode_length_buf += 1;
        // auto start = std::chrono::steady_clock::now();

        this->ComputeObservation();
        this->Forward();
        this->ComputeOutput();

        if (this->auto_log_pending && !this->log_data)
        {
            this->CSVInit(this->robot_name);
            this->log_data = true;
            this->auto_log_pending = false;
            std::cout << std::endl << LOGGER::INFO << "RL initialized - Auto data logging ENABLED (500Hz)" << std::endl;
        }

        output_dof_pos_queue.push(this->output_dof_pos);
        output_dof_vel_queue.push(this->output_dof_vel);
        output_dof_tau_queue.push(this->output_dof_tau);

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);


        // // ===== 3. 计算完整周期时间 =====
        // auto loop_end = std::chrono::steady_clock::now();

        // // 计算实际周期（毫秒）
        // double actual_period_ms;
        // if (first_loop) {
        //     actual_period_ms = 0.0;  // 首次循环无参考值
        //     first_loop = false;
        // } else {
        //     actual_period_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        //         loop_start - last_loop_end  // 本次开始与上次结束的时间差
        //     ).count() / 1000.0;
        // }

        // // ===== 4. 打印关键指标 =====
        // std::cout << std::fixed << std::setprecision(3);
        // std::cout << "Cycle Period: " << actual_period_ms << "ms | ";
        // std::cout << "Theoretical: 20.000ms | ";
        // std::cout << "Deviation: " << (actual_period_ms - 20.0) << "ms" << std::endl;

        // // ===== 5. 更新时间记录 =====
        // last_loop_end = loop_end;
    }
}

void RL_Real::Forward()
{
    this->active_model->Forward(this->params, this->onnx_tensor, this->obs);
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i) {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());

        if (i < 12) {
            this->plot_real_joint_pos[i].push_back(this->robot_state.motor_state.q[i]);
            this->plot_target_joint_pos[i].push_back(this->leg_command.q_ref[i]);
        } else {
            this->plot_real_joint_pos[i].push_back(this->robot_state.motor_state.q[i]);
            this->plot_target_joint_pos[i].push_back(this->arm_command.q_ref[i - 12]);
        }

        plt::subplot(this->params.num_of_dofs, 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

void RL_Real::InitLowCmd()
{
    this->leg_command.q_ref.setZero();
    this->leg_command.dq_ref.setZero();
    this->leg_command.kp.setZero();
    this->leg_command.kd.setZero();
    this->leg_command.tau_forward.setZero();

    this->arm_command.q_ref.setZero();
    this->arm_command.dq_ref.setZero();
    this->arm_command.kp.setZero();
    this->arm_command.kd.setZero();
    this->arm_command.tau_forward.setZero();
}
// 信号处理函数（仅设置退出标志，不做复杂操作）
void sigint_handler(int signum)
{
    if (signum == SIGINT) {
        RL_Real::exit_flag = true;  // 设置退出标志
        std::cout << "\n收到 Ctrl+C 准备退出..." << std::endl;
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, sigint_handler);

    RL_Real rl_sar;
    while (!RL_Real::exit_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 避免CPU占用过高
    }

    // 主循环退出后，rl_real 会离开作用域，自动调用析构函数
    return 0;
}
