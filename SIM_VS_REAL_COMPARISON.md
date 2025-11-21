# Sim vs Real 实现对比分析

## 🔍 关键差异：力矩计算和限制方式

### 1. SetCommand 实现对比

#### rl_sim.cpp (仿真版本) ✅ 正确
```cpp
// Line 271-287
for (int i = 0; i < this->params.num_of_dofs; ++i) {
    int mujoco_idx = this->params.joint_mapping[i];

    // 1️⃣ 先计算位置和速度误差
    double pos_err = command->motor_command.q[i] - this->mj_data->sensordata[mujoco_idx];
    double vel_err = 0.0 - this->mj_data->sensordata[mujoco_idx + this->params.num_of_dofs];
    
    // 2️⃣ 计算总力矩 = tau_forward + PD项
    double u = command->motor_command.tau[i] +              // 前馈力矩
               command->motor_command.kp[i] * pos_err +      // P项
               command->motor_command.kd[i] * vel_err;       // D项
   
    // 3️⃣ Clip 总力矩
    double lim = this->params.torque_limits[i];
    if (u >  lim) u =  lim;
    if (u < -lim) u = -lim;
    
    // 4️⃣ 下发clip后的总力矩
    this->mj_data->ctrl[mujoco_idx] = u;
}
```

**特点**：
- ✅ 在上层计算总力矩
- ✅ Clip总力矩
- ✅ 保证实际下发的力矩不超限

---

#### rl_real_atom.cpp (真机版本) ⚠️ 有问题
```cpp
// Line 293-307 (腿部)
for (int i = 0; i < 12; ++i) {
    // 1️⃣ 直接传递位置、速度、kp、kd给底层SDK
    this->leg_command.q_ref[i] = command->motor_command.q[i];
    this->leg_command.dq_ref[i] = command->motor_command.dq[i];
    this->leg_command.kp[i] = command->motor_command.kp[i];
    this->leg_command.kd[i] = command->motor_command.kd[i];
    
    // 2️⃣ 只clip前馈力矩 tau_forward
    double tau = command->motor_command.tau[i];
    double tau_limit = this->params.torque_limits[i];
    if (tau > tau_limit) tau = tau_limit;
    if (tau < -tau_limit) tau = -tau_limit;
    this->leg_command.tau_forward[i] = tau;
}

// 底层SDK计算: 
// 实际总力矩 = tau_forward + kp*(q_ref - q_actual) - kd*dq_actual
// ⚠️ 这个总力矩可能超限！
```

**问题**：
- ❌ 只clip了前馈力矩，没有clip总力矩
- ❌ PD项由底层SDK计算，上层无法控制
- ❌ 当位置误差或速度误差较大时，总力矩可能严重超限

---

### 2. CSV记录的力矩对比

#### rl_sim.cpp (仿真版本)
```cpp
// Line 572-577
#ifdef CSV_LOGGER
    vector_t tau_est = Eigen::Map<const vector_t>(
        this->robot_state.motor_state.tau_est.data(),
        this->robot_state.motor_state.tau_est.size()
    );        
    this->CSVLogger(this->output_dof_tau, tau_est, ...);
#endif
```

**记录内容**：
- `output_dof_tau`: 不清楚具体值（需要查看ComputeOutput）
- `tau_est`: 实际估计力矩（来自仿真器）

---

#### rl_real_atom.cpp (真机版本)
```cpp
// Line 495-510, 527
// 计算PD控制的力矩
vector_t cmd_tau_full = vector_t::Zero(logged_dofs);
if (control_dim > 0) {
    vector_t tau_cmd = this->params.rl_kp.head(control_dim).array() *
                       (this->output_dof_pos.segment(0, control_dim) - this->obs.dof_pos.segment(0, control_dim)).array()
                       - this->params.rl_kd.head(control_dim).array() *
                         this->obs.dof_vel.segment(0, control_dim).array();
    cmd_tau_full.segment(0, control_dim) = tau_cmd;
}

// Clip cmd_tau (已修复)
for(int i = 0; i < logged_dofs; ++i) {
    double tau_limit = this->params.torque_limits[i];
    if (cmd_tau_full[i] > tau_limit) cmd_tau_full[i] = tau_limit;
    if (cmd_tau_full[i] < -tau_limit) cmd_tau_full[i] = -tau_limit;
}

this->CSVLogger(joint_pos, joint_vel, tau_est, cmd_pos_full, cmd_tau_full, motion_phase);
```

**记录内容**：
- `cmd_tau_full`: **只是PD控制计算的力矩**（不包含前馈力矩tau_forward！）
- `tau_est`: 实际估计力矩（来自机器人传感器）

---

## ⚠️ 发现的严重问题

### 问题1: 真机的总力矩可能超限

**示例场景**（右肩Pitch，限制56 Nm）:
```
command->motor_command.tau[20] = 50 Nm (前馈力矩)
→ clip到 50 Nm (未超限)
→ 下发 tau_forward = 50 Nm

但在底层SDK中：
q_ref = -0.1 rad
q_actual = 0.2 rad  (位置误差 = -0.3 rad)
kp = 300
kd = 10
dq_actual = 0.5 rad/s

实际总力矩 = 50 + 300*(-0.3) - 10*0.5
          = 50 - 90 - 5
          = -45 Nm  (这种情况还好)

但如果误差反向：
q_ref = 0.2 rad
q_actual = -0.1 rad  (位置误差 = 0.3 rad)

实际总力矩 = 50 + 300*(0.3) - 10*0.5
          = 50 + 90 - 5
          = 135 Nm  ⚠️ 超限！(限制56 Nm)
```

### 问题2: CSV记录的cmd_tau不完整

**当前记录的计算**:
```cpp
tau_cmd = kp * (q_ref - q_actual) - kd * dq_actual
```

**但实际下发的是**:
```cpp
tau_forward (已clip) + kp + kd (在底层SDK计算)
```

**CSV中缺少 tau_forward 部分！**

---

## 📊 真机CSV数据的真实含义

查看CSV列：
```
cmd_tau_0, cmd_tau_1, ..., cmd_tau_26
```

**这些值实际是**:
```
cmd_tau[i] = kp[i] * (output_dof_pos[i] - dof_pos[i]) - kd[i] * dof_vel[i]
```

**缺少的部分**:
```
command->motor_command.tau[i]  (前馈力矩，未记录！)
```

**实际下发的总力矩**:
```
实际总力矩 = tau_forward + kp*(q_ref - q_actual) - kd*dq_actual
```

CSV只记录了PD项，**没有记录前馈项tau_forward**！

---

## 🔧 需要修复的问题

### 修复1: 让真机的力矩限制与仿真一致

**目标**: 在上层计算并clip总力矩，而不是只clip前馈力矩

**方案A**: 修改SetCommand，计算总力矩后再clip（推荐）
```cpp
// 在上层计算总力矩
double pos_err = command->motor_command.q[i] - state->motor_state.q[i];
double vel_err = 0.0 - state->motor_state.dq[i];

double u_total = command->motor_command.tau[i] +
                 command->motor_command.kp[i] * pos_err +
                 command->motor_command.kd[i] * vel_err;

// Clip总力矩
double lim = this->params.torque_limits[i];
if (u_total > lim) u_total = lim;
if (u_total < -lim) u_total = -lim;

// 方式1: 调整tau_forward，保持PD不变
this->leg_command.tau_forward[i] = u_total - (kp * pos_err + kd * vel_err);

// 或方式2: 只用tau_forward，kp=kd=0
this->leg_command.tau_forward[i] = u_total;
this->leg_command.kp[i] = 0.0;
this->leg_command.kd[i] = 0.0;
```

**方案B**: 依赖底层SDK的力矩限制（如果SDK有）
- 检查atom_sdk是否有总力矩限制功能
- 如果有，确保SDK的限制与config.yaml一致

---

### 修复2: CSV记录完整的力矩信息

**当前问题**: CSV只记录PD项，缺少tau_forward

**修改方案**: 将command->motor_command.tau也记录到CSV

**修改位置**: Line 495-527

**新增记录内容**:
```cpp
// 记录前馈力矩（clip前）
vector_t tau_forward_raw = vector_t::Zero(logged_dofs);
for(int i = 0; i < logged_dofs; ++i) {
    tau_forward_raw[i] = command->motor_command.tau[i];
}

// 记录前馈力矩（clip后）
vector_t tau_forward_clipped = vector_t::Zero(logged_dofs);
for(int i = 0; i < logged_dofs; ++i) {
    double tau = command->motor_command.tau[i];
    double tau_limit = this->params.torque_limits[i];
    if (tau > tau_limit) tau = tau_limit;
    if (tau < -tau_limit) tau = -tau_limit;
    tau_forward_clipped[i] = tau;
}

// 记录PD项
vector_t tau_pd = this->params.rl_kp.head(control_dim).array() *
                  (this->output_dof_pos.segment(0, control_dim) - this->obs.dof_pos.segment(0, control_dim)).array()
                  - this->params.rl_kd.head(control_dim).array() *
                    this->obs.dof_vel.segment(0, control_dim).array();

// 记录总力矩（clip前）
vector_t tau_total_raw = tau_forward_raw + tau_pd;

// 记录总力矩（clip后）
vector_t tau_total_clipped = vector_t::Zero(logged_dofs);
for(int i = 0; i < logged_dofs; ++i) {
    tau_total_clipped[i] = tau_forward_clipped[i] + tau_pd[i];
    // 再次clip总力矩
    double tau_limit = this->params.torque_limits[i];
    if (tau_total_clipped[i] > tau_limit) tau_total_clipped[i] = tau_limit;
    if (tau_total_clipped[i] < -tau_limit) tau_total_clipped[i] = -tau_limit;
}
```

**建议CSV列**:
- `tau_forward_raw`: 原始前馈力矩
- `tau_forward_clip`: clip后的前馈力矩
- `tau_pd`: PD控制力矩
- `tau_total_raw`: 总力矩（clip前）
- `tau_total_clip`: 总力矩（clip后）
- `tau_est`: 实际估计力矩（传感器读数）

---

## 📋 对比总结

| 项目 | rl_sim.cpp (仿真) | rl_real_atom.cpp (真机) | 问题 |
|------|-------------------|------------------------|------|
| **力矩计算位置** | 上层 | 底层SDK | ❌ 真机上层无法控制 |
| **Clip对象** | 总力矩 | 只clip前馈力矩 | ❌ 总力矩可能超限 |
| **PD计算** | 上层计算后clip | 底层SDK计算 | ❌ 无法保证总力矩限制 |
| **CSV记录** | output_dof_tau | 只记录PD项 | ❌ 缺少tau_forward |

---

## 🎯 推荐修改优先级

### 🔴 高优先级（安全问题）
1. **修复SetCommand**: 计算并clip总力矩，确保下发的力矩不超限
2. **验证底层SDK**: 检查atom_sdk是否有额外的力矩保护

### 🟡 中优先级（数据分析）
3. **完善CSV记录**: 记录完整的力矩分解（tau_forward + tau_pd + tau_total）
4. **添加日志**: 记录clip前后的力矩对比

### 🟢 低优先级（代码质量）
5. 统一sim和real的实现方式
6. 添加单元测试验证力矩限制功能

