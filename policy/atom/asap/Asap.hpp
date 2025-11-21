#pragma once
#include "rl_sdk.hpp"



// count: 2
// Shape: [1, 150]  obs
// Shape: [1, 1]    time_step

// 7个output
// actions  
// Shape: [1, 27]
// joint_pos
// Shape: [1, 27]
// joint_vel
// Shape: [1, 27]
// body_pos_w
// Shape: [1, 14, 3]
// body_quat_w
// Shape: [1, 14, 4]
// body_lin_vel_w
// Shape: [1, 14, 3]
// body_ang_vel_w
// Shape: [1, 14, 3]

class AsapModel : public RobotModel {

private:


    Ort::MemoryInfo memoryInfo; 
    int counter_step;
    double motion_phase;

public:
    AsapModel(): memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)){
        this->counter_step = 0;
    }
    
    // 重置策略状态（用于 Reset）
    void reset() {
        this->counter_step = 0;
        this->motion_phase = 0.0;
    }
    
    // Get current motion phase (0.0 to 1.0)
    double get_motion_phase() const {
        return this->motion_phase;
    }
    
    vector_t compute_observation(const ModelParams& params, const RobotState<double>& robot_state, const Control& control, Observations& obs){

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
        
        // 🔍 Debug: 打印前5步的详细观测数据
        if (this->counter_step < 5) {
            std::cout << "\n[C++ Step " << this->counter_step << " Observation Debug]" << std::endl;
            std::cout << "  quaternion (w,x,y,z): " << obs.base_quat.w() << ", " 
                      << obs.base_quat.x() << ", " << obs.base_quat.y() << ", " << obs.base_quat.z() << std::endl;
            std::cout << "  ang_vel: " << obs.ang_vel[0] << ", " << obs.ang_vel[1] << ", " << obs.ang_vel[2] << std::endl;
            std::cout << "  dof_pos[0-2]: " << obs.dof_pos[0] << ", " << obs.dof_pos[1] << ", " << obs.dof_pos[2] << std::endl;
            std::cout << "  dof_vel[0-2]: " << obs.dof_vel[0] << ", " << obs.dof_vel[1] << ", " << obs.dof_vel[2] << std::endl;
            std::cout << "  last_action[0-2]: " << obs.actions[0] << ", " << obs.actions[1] << ", " << obs.actions[2] << std::endl;
        }

      
        double raw_phase = params.dt * params.decimation * static_cast<double>(this->counter_step + 1) / params.motion_time;
        this->motion_phase = std::fmod(raw_phase, 1.0);  // ✅ 使用取模，让相位在 0-1 之间循环

        // 首次执行时打印完整参数
        static bool first_run = true;
        if (first_run) {
            std::cout << "\n========== FIRST OBSERVATION COMPUTATION ==========" << std::endl;
            std::cout << "[Params Check]" << std::endl;
            std::cout << "  dt = " << params.dt << " s" << std::endl;
            std::cout << "  decimation = " << params.decimation << std::endl;
            std::cout << "  motion_time = " << params.motion_time << " s" << std::endl;
            std::cout << "  dt * decimation = " << (params.dt * params.decimation) << " s" << std::endl;
            std::cout << "  RL frequency = " << (1.0 / (params.dt * params.decimation)) << " Hz" << std::endl;
            std::cout << "  ang_vel_scale = " << params.ang_vel_scale << std::endl;
            std::cout << "  dof_pos_scale = " << params.dof_pos_scale << std::endl;
            std::cout << "  dof_vel_scale = " << params.dof_vel_scale << std::endl;
            std::cout << "  action_scale (first 3): [" << params.action_scale[0] << ", " 
                      << params.action_scale[1] << ", " << params.action_scale[2] << "]" << std::endl;
            std::cout << "===================================================" << std::endl;
            first_run = false;
        }
       
        if (this->counter_step % 100 == 0) {
            std::cout << "\n[Phase Debug] counter_step=" << this->counter_step 
                      << ", motion_phase=" << this->motion_phase
                      << ", dt*decimation=" << (params.dt * params.decimation)
                      << ", motion_time=" << params.motion_time << std::endl;
        }

        vector_t deltaJointPos = obs.dof_pos - params.default_dof_pos;

        vector_t proprioObs(params.num_one_step_observations);
        
        vector_t projectedGravity = QuatRotateInverse(obs.base_quat, obs.gravity_vec);
        
        // 🔍 Debug: 打印projected gravity计算
        if (this->counter_step < 5) {
            std::cout << "  projected_gravity: " << projectedGravity[0] << ", " 
                      << projectedGravity[1] << ", " << projectedGravity[2] << std::endl;
            std::cout << "  deltaJointPos[0-2]: " << deltaJointPos[0] << ", " 
                      << deltaJointPos[1] << ", " << deltaJointPos[2] << std::endl;
        }

        double gvec_scale = 1.0;
        double refmotion_scale = 1.0;
        
        proprioObs << obs.actions,
            obs.ang_vel * params.ang_vel_scale, 
            deltaJointPos * params.dof_pos_scale, 
            obs.dof_vel * params.dof_vel_scale, 
            projectedGravity * gvec_scale, 
            this->motion_phase * refmotion_scale;  

        // update motion phase
        this-> counter_step += 1;
        return proprioObs;
    }

    void Forward(const ModelParams& params, OnnxTensor& onnx_tensor, Observations& obs){
        
        // create input tensor object
        std::vector<Ort::Value> inputValues;
        // 输入 1: obs 
        inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(this->memoryInfo, 
                                                                    obs.observations.data(), 
                                                                    obs.observations.size(),
                                                                    onnx_tensor.inputShapes[0].data(),
                                                                    onnx_tensor.inputShapes[0].size()));
        
        // run inference
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputValues = onnx_tensor.sessionPtr->Run(
            runOptions, 
            onnx_tensor.inputNames.data(), 
            inputValues.data(), 
            1, 
            onnx_tensor.outputNames.data(), 
            1
        );

       
        for (int i = 0; i < params.action_dim; i++) { 
            obs.actions[i] =  *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i); 
        }

        scalar_t actionMin = -params.clip_actions;
        scalar_t actionMax = params.clip_actions;
        obs.actions = obs.actions.array().max(actionMin).min(actionMax);
    }
    
    std::string name() const override { return "Asap"; }
};
