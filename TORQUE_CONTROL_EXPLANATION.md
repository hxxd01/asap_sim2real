# 力矩控制详解：总力矩、下发力矩和kp/kd的关系

## 📊 完整的数据流

### 1. RL策略输出 → FSM设置command

**位置**: `policy/atom/fsm.hpp` Line 283-306 (RLFSMStateRL_Vel_Locomotion)

```cpp
void Run() override {
    // 从RL输出队列获取目标位置和速度
    vector_t _output_dof_pos, _output_dof_vel;
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && 
        rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        for (int i = 0; i < rl.params.action_dim; ++i)
        {
            // 1️⃣ 设置目标位置（RL策略输出）
            fsm_command->motor_command.q[i] = _output_dof_pos[i];
            
            // 2️⃣ 设置目标速度（RL策略输出）
            fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            
            // 3️⃣ 设置PD增益（从config.yaml读取）
            fsm_command->motor_command.kp[i] = rl.params.rl_kp[i];
            fsm_command->motor_command.kd[i] = rl.params.rl_kd[i];
            
            // 4️⃣ ⚠️ 前馈力矩设为0！
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}
```

**关键发现**：
- ✅ `q[i]` = RL输出的目标位置
- ✅ `dq[i]` = RL输出的目标速度
- ✅ `kp[i]`, `kd[i]` = 从config.yaml读取的PD增益
- ⚠️ **`tau[i] = 0`** (前馈力矩始终为0！)

---

### 2. SetCommand: 计算并下发力矩

#### 仿真版本 (rl_sim.cpp)

```cpp
void RL_Sim::SetCommand(const RobotCommand<double> *command, const RobotState<double> *state)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        // 1️⃣ 计算位置和速度误差
        double pos_err = command->motor_command.q[i] - this->mj_data->sensordata[mujoco_idx];
        double vel_err = 0.0 - this->mj_data->sensordata[mujoco_idx + this->params.num_of_dofs];
        
        // 2️⃣ 计算总力矩 = 前馈力矩 + PD项
        //    注意：command->motor_command.tau[i] = 0 (从FSM设置)
        double u = command->motor_command.tau[i] +              // = 0
                   command->motor_command.kp[i] * pos_err +      // PD项
                   command->motor_command.kd[i] * vel_err;       // PD项
        
        // 3️⃣ Clip总力矩
        double lim = this->params.torque_limits[i];
        if (u >  lim) u =  lim;
        if (u < -lim) u = -lim;
        
        // 4️⃣ 下发clip后的总力矩
        this->mj_data->ctrl[mujoco_idx] = u;
    }
}
```

**仿真中的总力矩**：
```
总力矩 = 0 + kp*(q_ref - q_actual) + kd*(dq_ref - dq_actual)
       = kp*pos_err + kd*vel_err
```

---

#### 真机版本 (rl_real_atom.cpp)

```cpp
void RL_Real::SetCommand(const RobotCommand<double> *command, const RobotState<double> *state)
{
    for (int i = 0; i < 12; ++i) {
        // 1️⃣ 直接传递参数给底层SDK
        this->leg_command.q_ref[i] = command->motor_command.q[i];
        this->leg_command.dq_ref[i] = command->motor_command.dq[i];
        this->leg_command.kp[i] = command->motor_command.kp[i];
        this->leg_command.kd[i] = command->motor_command.kd[i];
        
        // 2️⃣ 只clip前馈力矩（但tau=0，所以clip后还是0）
        double tau = command->motor_command.tau[i];  // = 0
        double tau_limit = this->params.torque_limits[i];
        if (tau > tau_limit) tau = tau_limit;
        if (tau < -tau_limit) tau = -tau_limit;
        this->leg_command.tau_forward[i] = tau;  // = 0
    }
    
    // 3️⃣ 底层SDK计算总力矩
    // 实际总力矩 = tau_forward + kp*(q_ref - q_actual) - kd*dq_actual
    //            = 0 + kp*pos_err - kd*dq_actual
}
```

**真机中的总力矩**：
```
实际总力矩 = 0 + kp*(q_ref - q_actual) - kd*dq_actual
          = kp*pos_err - kd*dq_actual
```

---

## 🔍 关键问题解答

### Q1: 总力矩、下发力矩和kp/kd有啥区别？

| 概念 | 定义 | 在代码中的位置 |
|------|------|---------------|
| **前馈力矩 (tau_forward)** | RL策略直接输出的力矩 | `command->motor_command.tau[i]` = **0** |
| **PD控制项** | 基于位置/速度误差的反馈力矩 | `kp*pos_err + kd*vel_err` |
| **总力矩** | 前馈 + PD项 | `tau_forward + kp*pos_err + kd*vel_err` |
| **下发力矩** | 实际发送给执行器的力矩 | 仿真：clip后的总力矩<br>真机：底层SDK计算的总力矩 |

**当前实现**：
- 前馈力矩 = 0（FSM中设置）
- 总力矩 = PD项（因为tau=0）
- 下发力矩 = 总力矩（仿真中clip后，真机中由SDK计算）

---

### Q2: 为什么计算下发力矩要用kp/kd？

**答案**：因为当前实现是**纯PD位置控制**，没有前馈力矩！

**控制公式**：
```
下发力矩 = tau_forward + kp*(q_ref - q_actual) + kd*(dq_ref - dq_actual)
         = 0 + kp*pos_err + kd*vel_err
         = kp*pos_err + kd*vel_err
```

**为什么需要PD控制**：
1. **P项 (kp)**: 提供位置误差的恢复力
   - 如果实际位置 < 目标位置 → 产生正向力矩
   - 如果实际位置 > 目标位置 → 产生负向力矩
   - 使关节趋向目标位置

2. **D项 (kd)**: 提供速度误差的阻尼力
   - 如果实际速度 > 目标速度 → 产生负向力矩（减速）
   - 如果实际速度 < 目标速度 → 产生正向力矩（加速）
   - 减少振荡，提高稳定性

**示例**：
```
目标位置 q_ref = 0.5 rad
实际位置 q_actual = 0.3 rad
位置误差 pos_err = 0.2 rad
kp = 300

P项力矩 = 300 * 0.2 = 60 Nm  (产生正向力矩，推动关节向0.5 rad移动)
```

---

### Q3: 为什么最后算总力矩也用了kp/kd？

**答案**：因为总力矩 = 前馈力矩 + PD项，而前馈力矩=0，所以总力矩=PD项

**完整公式**：
```
总力矩 = tau_forward + kp*pos_err + kd*vel_err
       = 0 + kp*pos_err + kd*vel_err
       = kp*pos_err + kd*vel_err
```

**为什么需要计算总力矩**：
1. **力矩限制保护**：需要知道总力矩才能clip到安全范围
2. **与仿真一致**：仿真中也是计算总力矩后clip
3. **安全考虑**：即使前馈力矩=0，PD项也可能超限

**示例（右肩Pitch，限制56 Nm）**：
```
q_ref = 0.2 rad
q_actual = -0.1 rad
pos_err = 0.3 rad
kp = 300
kd = 2
dq_actual = 0.5 rad/s

总力矩 = 0 + 300*0.3 + 2*(-0.5)
       = 90 - 1
       = 89 Nm  ⚠️ 超限！(限制56 Nm)
```

---

## 📋 当前实现的问题

### ⚠️ 问题1: 真机总力矩可能超限

**原因**：
- 真机只clip了前馈力矩（但tau=0，所以没意义）
- 底层SDK计算的PD项可能很大
- 总力矩 = 0 + PD项，可能超限

**解决方案**：
在上层计算总力矩并clip，然后以纯力矩模式下发（kp=kd=0）

---

### ⚠️ 问题2: CSV记录的cmd_tau不准确

**当前CSV记录** (Line 495-510):
```cpp
// 只计算了PD项
cmd_tau = kp*(q_ref - q) - kd*dq
```

**问题**：
- 缺少前馈力矩（虽然tau=0，但应该记录）
- 没有记录clip后的总力矩
- 无法验证力矩限制是否生效

---

## 🎯 总结

### 当前控制模式：纯PD位置控制

```
RL策略输出 → 目标位置 q_ref
            ↓
FSM设置 → q_ref, dq_ref, kp, kd, tau=0
            ↓
SetCommand → 计算总力矩 = 0 + kp*pos_err + kd*vel_err
            ↓
仿真：clip总力矩后下发
真机：下发kp/kd给SDK，SDK计算总力矩（可能超限！）
```

### 关键点

1. **前馈力矩始终为0**：FSM中 `tau[i] = 0`
2. **总力矩 = PD项**：因为tau=0
3. **需要kp/kd的原因**：提供位置和速度的反馈控制
4. **计算总力矩的原因**：需要clip保护，防止超限

### 建议

1. **修复真机力矩限制**：在上层计算并clip总力矩
2. **完善CSV记录**：记录完整的力矩分解
3. **考虑添加前馈力矩**：如果RL策略需要直接输出力矩

