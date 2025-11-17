#pragma once
#include "rl_sdk.hpp"

class HomieModel : public RobotModel {

private:
    double gait_indices = 0.0;
    std::vector<double> foot_indices = {0.0, 0.0}; // FL, FR
    std::vector<double> clock_inputs = {0.0, 0.0};
    vector_t desired_contact_states;

    Ort::MemoryInfo memoryInfo;

    double normalCDF(double x, double sigma = 0.07) {
        return 0.5 * (1 + std::erf(x / (sigma * std::sqrt(2))));
    }

    double smoothPhase(double phase) {
        phase = fmod(phase, 1.0);
        return normalCDF(phase) * (1 - normalCDF(phase - 0.5)) + 
            normalCDF(phase - 1) * (1 - normalCDF(phase - 1.5));
    }

public:
    HomieModel(): memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)){
        // constructor implementation (if needed)
        this->desired_contact_states.resize(2);
        this->desired_contact_states.setZero();
    }

    vector_t get_contact_state(){
        return this->desired_contact_states;
    }

    void step_contact_targets(const ModelParams& params, double frequencies) {

        this->gait_indices = fmod(this->gait_indices + params.dt * params.decimation * frequencies, 1.0);
        
        double phase_offset = 0.5;

        this->foot_indices[0] = fmod(this->gait_indices + phase_offset, 1.0); // FL
        this->foot_indices[1] = fmod(this->gait_indices, 1.0);  // FR

        const double stance_duration = 0.5; // 50%

        for(int i = 0; i < 2; ++i) {
            if (this->foot_indices[i] < stance_duration + 1e-6) {
                this->foot_indices[i] =  this->foot_indices[i] *(0.5 / stance_duration); 
            } else {
            this->foot_indices[i] = 0.5 + (this->foot_indices[i] - stance_duration) * (0.5 / (1.0 - stance_duration)); 
            }

            this->clock_inputs[i] = sin(2 * M_PI * this->foot_indices[i]);
            this->desired_contact_states[i] = smoothPhase(foot_indices[i]);
        }
    }
    
    vector_t compute_observation(const ModelParams& params, const RobotState<double>& robot_state, const Control& control, Observations& obs){

        vector_t command(5);
        command << control.x, control.y, control.yaw, control.height, control.frequency;

        step_contact_targets(params, control.frequency);

        bool is_standing = command.head(3).norm() < 0.1;

        vector_t processed_clock_inputs(clock_inputs.size());

        if (is_standing) {
            processed_clock_inputs.setOnes();
        } else {
            processed_clock_inputs = vector_t::Map(clock_inputs.data(), clock_inputs.size());
        }
        for(int i = 0; i < params.num_of_policy_dofs; i++)
        {
            obs.dof_pos[i] = robot_state.motor_state.q[i];
            obs.dof_vel[i] = robot_state.motor_state.dq[i];
        }

        obs.base_quat.w() = robot_state.imu.quaternion[0];
        obs.base_quat.x() = robot_state.imu.quaternion[1];
        obs.base_quat.y() = robot_state.imu.quaternion[2];
        obs.base_quat.z() = robot_state.imu.quaternion[3];

        for (int i = 0; i < 3; i++)
        {
            obs.ang_vel[i] = robot_state.imu.gyroscope[i];
        }
        vector_t projectedGravity = QuatRotateInverse(obs.base_quat, obs.gravity_vec);

        vector_t deltaJointPos = obs.dof_pos - params.default_dof_pos;

        vector_t proprioObs(params.num_one_step_observations);

        if(params.num_one_step_observations == 79)
            proprioObs << command.head(3).array() * params.commands_scale.array(), 
                command.segment(3, 2),  // 修复为从索引3取2个元素
                obs.ang_vel * params.ang_vel_scale, 
                projectedGravity, 
                deltaJointPos * params.dof_pos_scale, 
                obs.dof_vel * params.dof_vel_scale, 
                obs.actions,
                processed_clock_inputs;
        else
            proprioObs << command.head(3).array() * params.commands_scale.array(), 
                command.segment(3, 1),  // 修复为从索引3取2个元素
                obs.ang_vel * params.ang_vel_scale, 
                projectedGravity, 
                deltaJointPos * params.dof_pos_scale, 
                obs.dof_vel * params.dof_vel_scale, 
                obs.actions;

        return proprioObs;
    }

    void Forward(const ModelParams& params, OnnxTensor& onnx_tensor, Observations& obs){
        
        // std::vector<tensor_element_t> actions; 
        // actions.resize(params.action_dim);
        // create input tensor object
        std::vector<Ort::Value> inputValues;
        inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(this->memoryInfo, obs.observations.data(), obs.observations.size(),
                                                                            onnx_tensor.inputShapes[0].data(), onnx_tensor.inputShapes[0].size()));
        // run inference
        Ort::RunOptions runOptions;
// #ifdef KALMAN_FILTER
//         std::vector<Ort::Value> outputValues = onnx_tensor.sessionPtr->Run(runOptions, onnx_tensor.inputNames.data(), inputValues.data(), 1, onnx_tensor.outputNames.data(), 2);
// #else
        std::vector<Ort::Value> outputValues = onnx_tensor.sessionPtr->Run(runOptions, onnx_tensor.inputNames.data(), inputValues.data(), 1, onnx_tensor.outputNames.data(), 1);
// #endif 

        for (int i = 0; i < params.action_dim; i++) { obs.actions[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i); }
// #ifdef KALMAN_FILTER
//         for (int i = 0; i < 3; i++)
//         {   
//             obs.network_est_lin_vel[i] = *(outputValues[1].GetTensorMutableData<tensor_element_t>() + i);
//         }
// #endif
        scalar_t actionMin = -params.clip_actions;
        scalar_t actionMax = params.clip_actions;
        obs.actions = obs.actions.array().max(actionMin).min(actionMax);
    }
    
    std::string name() const override { return "Homie"; }
};
