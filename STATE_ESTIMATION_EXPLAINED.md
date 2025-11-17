# FSM 状态估计的作用说明

## 📌 概述

在这个项目中，**状态估计（State Estimation）** 和 **FSM（有限状态机）** 是两个不同但相互配合的模块：

- **FSM**: 管理机器人的**控制模式**（如被动模式、站立、行走等）
- **状态估计**: 估计机器人的**物理状态**（位置、速度、姿态等）

## 🔍 状态估计的核心作用

### 1. **提供准确的机器人状态信息**

状态估计器（卡尔曼滤波器）的主要任务是融合多个传感器数据，提供更准确的机器人状态：

```cpp
// 在 rl_sim.cpp 的 RobotControl() 中
this->est_robot_state = this->state_estimator.Update(this->params.dt, &this->robot_state);
```

**估计的状态包括：**
- **基座位置** (x, y, z) - 6维状态的前3维
- **基座姿态** (roll, pitch, yaw) - 6维状态的4-6维
- **基座线速度** (vx, vy, vz) - 用于 `obs.est_lin_vel`
- **基座角速度** (wx, wy, wz)
- **关节位置** (12个关节)
- **关节速度** (12个关节)

### 2. **为什么需要状态估计？**

#### 问题：直接测量不准确

在真实机器人中，某些状态**无法直接准确测量**：

1. **基座线速度**：
   - ❌ 无法直接测量（没有速度传感器）
   - ✅ 需要通过 IMU 和足端接触信息融合估计

2. **基座位置**：
   - ❌ GPS 在室内不可用
   - ✅ 需要通过积分和接触约束估计

3. **传感器噪声**：
   - IMU 有漂移和噪声
   - 关节编码器有测量误差
   - 需要融合多个传感器提高精度

#### 解决方案：卡尔曼滤波

状态估计器使用**扩展卡尔曼滤波（EKF）**融合：
- **IMU 数据**（加速度计、陀螺仪）
- **关节编码器**（位置、速度）
- **足端接触信息**（接触力、接触状态）
- **运动学模型**（Pinocchio）

### 3. **在 RL 控制中的关键作用**

#### 观测空间构建

状态估计的结果直接用于构建 RL 策略的**观测向量**：

```cpp
// 在 rl_sim.cpp 中
for (int i = 0; i < 3; i++)
{
    this->obs.est_lin_vel[i] = this->est_robot_state[6+i];  // 使用估计的线速度
    file << this->obs.lin_vel[i] << ",";  // 记录真实值用于对比
}
```

**观测向量包含：**
- `obs.est_lin_vel`: **估计的基座线速度**（用于 RL 策略）
- `obs.lin_vel`: 真实/模拟的线速度（用于记录和调试）

#### 为什么 RL 策略需要估计的速度？

1. **训练一致性**：
   - RL 策略在仿真中训练时，使用的是**估计的速度**（模拟真实机器人）
   - 部署到真实机器人时，也使用**估计的速度**
   - 保持训练和部署的一致性

2. **状态可观测性**：
   - 真实机器人无法直接测量基座速度
   - 必须通过状态估计获得
   - 这是 RL 策略能正常工作的前提

3. **鲁棒性**：
   - 估计值融合了多个传感器信息
   - 比单一传感器更可靠
   - 减少传感器故障的影响

## 🔄 FSM 和状态估计的协作

### FSM 状态转换流程

```
┌─────────────────┐
│  Passive 状态   │  ← 初始状态，机器人被动
└────────┬────────┘
         │ 按 Num0/A 键
         ▼
┌─────────────────┐
│  GetUp 状态     │  ← 机器人站起
└────────┬────────┘
         │ 站起完成
         ▼
┌─────────────────┐
│ RL_Locomotion   │  ← **启用状态估计和 RL 控制**
└─────────────────┘
```

### 在 RL 状态中的状态估计

在 `RLFSMStateRL_Locomotion` 状态中：

```cpp
void Run() override {
    // 1. 获取原始传感器数据
    GetState(&robot_state);
    
    // 2. **状态估计器更新**（关键步骤）
    if (rl_init_done) {
        est_robot_state = state_estimator.Update(dt, &robot_state);
        
        // 3. 将估计值用于观测
        obs.est_lin_vel[i] = est_robot_state[6+i];
    }
    
    // 4. RL 策略使用估计的观测进行推理
    // 5. 发送控制命令
}
```

## 📊 状态估计的数据流

```
传感器数据
    │
    ├─→ IMU (加速度、角速度)
    ├─→ 关节编码器 (位置、速度)
    └─→ 足端力传感器 (接触力)
         │
         ▼
┌────────────────────┐
│  卡尔曼滤波器      │
│  (状态估计器)      │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  est_robot_state   │
│  - 基座位置/姿态   │
│  - 基座速度        │
│  - 关节状态        │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  RL 观测向量       │
│  obs.est_lin_vel   │
│  (用于策略推理)     │
└────────────────────┘
```

## 🎯 具体应用场景

### 场景 1: 真实机器人部署

在真实机器人上，状态估计是**必需的**：

```cpp
// rl_real_atom.cpp
#if KALMAN_FILTER
    state_estimator.Update(dt, &robot_state);  // 必须使用状态估计
    obs.est_lin_vel = est_robot_state[6:9];    // 估计速度用于 RL
#endif
```

**原因：**
- 真实机器人没有直接的速度传感器
- 必须通过状态估计获得速度信息
- RL 策略依赖这个估计值

### 场景 2: 仿真模式

在仿真中，状态估计用于**模拟真实环境**：

```cpp
// rl_sim.cpp
// 即使有真实值，也使用估计值，保持一致性
obs.est_lin_vel[i] = est_robot_state[6+i];  // 使用估计值
// obs.lin_vel[i] = mj_data->sensordata[...];  // 真实值仅用于记录
```

**原因：**
- 保持仿真和真实部署的一致性
- 验证状态估计器的性能
- 调试和优化状态估计算法

## 🔧 状态估计器配置

配置文件：`config/kalman_config/atom/kalman_config.yaml`

```yaml
atom:
  imuProcessNoisePosition: 0.04      # IMU 位置过程噪声
  imuProcessNoiseVelocity: 0.02       # IMU 速度过程噪声
  footProcessNoisePosition: 0.02      # 足端位置过程噪声
  footSensorNoisePosition: 0.02       # 足端传感器噪声
  footSensorNoiseVelocity: 10.0       # 足端速度传感器噪声
  contact_threshold: 250.0           # 接触力阈值
```

**参数调优：**
- 噪声参数越大 → 越不相信该传感器
- 需要根据实际传感器性能调整
- 影响估计的精度和响应速度

## 📈 状态估计的输出

状态估计器输出 `est_robot_state`，维度为 **36**：

```
[0:6]   基座位置和姿态 (x, y, z, roll, pitch, yaw)
[6:9]   基座线速度 (vx, vy, vz)  ← **用于 obs.est_lin_vel**
[9:12]  基座角速度 (wx, wy, wz)
[12:24] 关节位置 (12个关节)
[24:36] 关节速度 (12个关节)
```

## ⚠️ 注意事项

### 1. **状态估计必须在 RL 控制之前初始化**

```cpp
// 在 RL_Sim 构造函数中
this->state_estimator = KalmanFilterEstimate(robot_name);
// 必须在 rl_init_done = true 之前完成
```

### 2. **状态估计需要接触信息**

```cpp
// 可选：更新接触状态（如果策略提供）
auto desired_contact = active_model->get_contact_state();
// state_estimator.updateContact(desired_contact);  // 当前被注释
```

### 3. **状态估计的重置**

```cpp
// 按 Y 键可以重置状态估计器
if (control.current_gamepad == Input::Gamepad::Y) {
    state_estimator.ResetStateEst();
}
```

## 🎓 总结

**FSM 状态估计的作用：**

1. ✅ **提供准确的机器人状态**：特别是无法直接测量的基座速度
2. ✅ **融合多传感器数据**：提高状态估计的鲁棒性
3. ✅ **支持 RL 策略**：为策略提供可靠的观测输入
4. ✅ **保持一致性**：仿真和真实部署使用相同的状态估计方法
5. ✅ **提高鲁棒性**：减少传感器噪声和故障的影响

**简单来说：**
- **FSM** 决定机器人**做什么**（控制模式）
- **状态估计** 告诉机器人**在哪里**（物理状态）
- 两者配合，实现稳定可靠的机器人控制

---

**相关文件：**
- 状态估计器实现：`include/kalman_estimator/LinearKalmanFilter.h`
- 状态估计器使用：`src/rl_sim.cpp` (第666行)
- 配置文件：`config/kalman_config/atom/kalman_config.yaml`

