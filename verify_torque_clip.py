#!/usr/bin/env python3
"""
验证力矩 clip 功能的效果
模拟原始 CSV 数据经过 clip 后的效果
"""

import pandas as pd
import numpy as np

# 力矩限制（从config.yaml）
torque_limits = [
    280.0, 308.0, 140.0, 360.0, 130.0, 130.0,  # 0-5: 左腿
    280.0, 308.0, 140.0, 360.0, 130.0, 130.0,  # 6-11: 右腿
    250.0,  # 12: 腰部
    56.0, 36.0, 36.0, 36.0, 18.6, 18.6, 18.6,  # 13-19: 左臂
    56.0, 36.0, 36.0, 36.0, 18.6, 18.6, 18.6,  # 20-26: 右臂
]

joint_names = [
    "左髋Roll", "左髋Pitch", "左膝", "左踝Pitch", "左踝Roll", "左踝Yaw",
    "右髋Roll", "右髋Pitch", "右膝", "右踝Pitch", "右踝Roll", "右踝Yaw",
    "腰部",
    "左肩Pitch", "左肩Roll", "左肘", "左腕1", "左腕2", "左腕3", "左手指",
    "右肩Pitch", "右肩Roll", "右肘", "右腕1", "右腕2", "右腕3", "右手指",
]

csv_file = "/home/dobot/Desktop/rl_deploy-develop-asap1/policy/atom/motor_20251120191132.csv"
df = pd.read_csv(csv_file)

print("="*100)
print("力矩 Clip 功能验证")
print("="*100)

# 提取cmd_tau列
cmd_tau_cols = [f'cmd_tau_{i}' for i in range(27)]
torques_original = df[cmd_tau_cols].values  # 原始力矩
limits = np.array(torque_limits)

# 应用 clip（模拟代码中的逻辑）
torques_clipped = np.clip(torques_original, -limits[np.newaxis, :], limits[np.newaxis, :])

# 计算差异
diff = torques_original - torques_clipped
clipped_mask = (diff != 0)

print(f"\n数据总数: {len(df)} 样本 × 27 关节 = {len(df) * 27} 数据点")
print(f"被 clip 的数据点: {np.sum(clipped_mask)} ({np.sum(clipped_mask) / (len(df) * 27) * 100:.2f}%)")

print("\n" + "="*100)
print("Clip 前后对比（仅显示有 clip 的关节）")
print("="*100)

for i in range(27):
    if np.any(clipped_mask[:, i]):
        original_abs_max = np.max(np.abs(torques_original[:, i]))
        clipped_abs_max = np.max(np.abs(torques_clipped[:, i]))
        num_clipped = np.sum(clipped_mask[:, i])
        
        # 找出最大的 clip 值
        max_clip_idx = np.argmax(np.abs(diff[:, i]))
        max_clip_original = torques_original[max_clip_idx, i]
        max_clip_clipped = torques_clipped[max_clip_idx, i]
        max_clip_diff = diff[max_clip_idx, i]
        
        print(f"\n关节 {i}: {joint_names[i]}")
        print(f"  限制值: ±{torque_limits[i]:.1f} Nm")
        print(f"  Clip 前最大绝对值: {original_abs_max:.2f} Nm")
        print(f"  Clip 后最大绝对值: {clipped_abs_max:.2f} Nm")
        print(f"  被 clip 的数据点: {num_clipped} / {len(df)} ({num_clipped/len(df)*100:.2f}%)")
        print(f"  最大 clip 量: {max_clip_diff:.2f} Nm ({max_clip_original:.2f} → {max_clip_clipped:.2f} Nm)")

print("\n" + "="*100)
print("\n总结:")
print(f"✅ Clip 功能会将所有超限力矩限制在 [-limit, limit] 范围内")
print(f"✅ 共有 {np.sum(clipped_mask)} 个数据点会被 clip")
print(f"✅ 右肩Pitch（关节20）的 -127.25 Nm 会被 clip 到 -56.0 Nm")
print(f"✅ 右肘（关节22）的 -64.25 Nm 会被 clip 到 -36.0 Nm")
print(f"✅ 右肩Roll（关节21）的 -57.63 Nm 会被 clip 到 -36.0 Nm")

print("\n" + "="*100)
print("\n注意:")
print("⚠️  此 clip 仅限制前馈力矩 tau_forward")
print("⚠️  实际总力矩 = tau_forward + kp*(q_ref - q) - kd*dq")
print("⚠️  总力矩仍可能超限，建议在底层SDK也添加保护")
print("="*100)

