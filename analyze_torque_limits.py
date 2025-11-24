#!/usr/bin/env python3
"""
分析CSV文件中的力矩是否超过限制
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

# 力矩限制（从config.yaml）
torque_limits = [
    280.0, 308.0, 140.0, 360.0, 130.0, 130.0,  # 0-5: 左腿
    280.0, 308.0, 140.0, 360.0, 130.0, 130.0,  # 6-11: 右腿
    250.0,  # 12: 腰部
    56.0, 36.0, 36.0, 36.0, 18.6, 18.6, 18.6,  # 13-19: 左臂
    56.0, 36.0, 36.0, 36.0, 18.6, 18.6, 18.6,  # 20-26: 右臂
]

def analyze_torque_limits(csv_file, torque_limits):
    """分析力矩是否超限"""
    print(f"正在读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    num_joints = len(torque_limits)
    print(f"关节数量: {num_joints}")
    print(f"数据行数: {len(df)}")
    
    # 提取cmd_tau列
    cmd_tau_cols = [f'cmd_tau_{i}' for i in range(num_joints)]
    
    # 检查列是否存在
    missing_cols = [col for col in cmd_tau_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺少列: {missing_cols}")
        return
    
    # 提取力矩数据
    torques = df[cmd_tau_cols].values  # shape: (n_samples, num_joints)
    
    # 转换为numpy数组以便计算
    limits = np.array(torque_limits)
    
    # 计算每个关节的力矩绝对值
    abs_torques = np.abs(torques)
    
    # 检查超限
    exceeded = abs_torques > limits[np.newaxis, :]
    
    # 统计信息
    print("\n" + "="*80)
    print("力矩超限分析结果")
    print("="*80)
    
    # 采样率（500Hz）
    sample_rate = 500.0  # Hz
    dt = 1.0 / sample_rate  # 0.002秒
    
    # 每个关节的超限统计
    print("\n各关节超限统计:")
    print("-" * 100)
    print(f"{'关节ID':<8} {'关节名称':<15} {'限制值(Nm)':<12} {'最大力矩(Nm)':<15} {'超限次数':<10} {'超限率(%)':<10} {'首次超限时间(s)':<15}")
    print("-" * 100)
    
    joint_names = [
    # 左腿 (0-5)
    "左髋Pitch", "左髋Roll", "左髋Yaw", "左膝", "左踝Pitch", "左踝Roll",
    # 右腿 (6-11)
    "右髋Pitch", "右髋Roll", "右髋Yaw", "右膝", "右踝Pitch", "右踝Roll",
    # 腰部 (12)
    "腰部Yaw",
    # 左臂 (13-19)
    "左肩Pitch", "左肩Roll", "左肩Yaw", "左肘Pitch", "左肘Roll", "左腕Pitch", "左腕Yaw",
    # 右臂 (20-26)
    "右肩Pitch", "右肩Roll", "右肩Yaw", "右肘Pitch", "右肘Roll", "右腕Pitch", "右腕Yaw",
]
    
    total_exceeded = 0
    max_exceeded_joint = -1
    max_exceeded_count = 0
    first_exceeded_time_overall = None  # 整体第一次超限时间
    
    for i in range(num_joints):
        exceeded_count = np.sum(exceeded[:, i])
        exceeded_rate = (exceeded_count / len(df)) * 100
        max_torque = np.max(abs_torques[:, i])
        
        # 找到第一次超限的时间
        first_exceeded_idx = None
        first_exceeded_time = None
        if exceeded_count > 0:
            first_exceeded_idx = np.where(exceeded[:, i])[0][0]  # 第一个超限的行索引
            first_exceeded_time = first_exceeded_idx * dt
            # 更新整体第一次超限时间
            if first_exceeded_time_overall is None or first_exceeded_time < first_exceeded_time_overall:
                first_exceeded_time_overall = first_exceeded_time
        
        joint_name = joint_names[i] if i < len(joint_names) else f"Joint_{i}"
        
        if first_exceeded_time is not None:
            print(f"{i:<8} {joint_name:<15} {torque_limits[i]:<12.1f} {max_torque:<15.2f} {exceeded_count:<10} {exceeded_rate:<10.2f} {first_exceeded_time:<15.3f}")
        else:
            print(f"{i:<8} {joint_name:<15} {torque_limits[i]:<12.1f} {max_torque:<15.2f} {exceeded_count:<10} {exceeded_rate:<10.2f} {'无超限':<15}")
        
        total_exceeded += exceeded_count
        if exceeded_count > max_exceeded_count:
            max_exceeded_count = exceeded_count
            max_exceeded_joint = i
    
    print("-" * 100)
    print(f"{'总计':<8} {'':<15} {'':<12} {'':<15} {total_exceeded:<10} {'':<10} {'':<15}")
    
    # 整体统计
    print("\n整体统计:")
    print("-" * 80)
    print(f"总数据点数: {len(df)}")
    print(f"总超限次数: {total_exceeded}")
    print(f"超限数据点占比: {(total_exceeded / (len(df) * num_joints)) * 100:.2f}%")
    print(f"至少有一个关节超限的样本数: {np.sum(np.any(exceeded, axis=1))}")
    print(f"所有关节同时超限的样本数: {np.sum(np.all(exceeded, axis=1))}")
    if first_exceeded_time_overall is not None:
        print(f"首次超限时间: {first_exceeded_time_overall:.3f} 秒 (行号: {int(first_exceeded_time_overall / dt) + 2})")
    else:
        print(f"首次超限时间: 无超限")
    
    # 最严重的关节
    if max_exceeded_joint >= 0:
        print(f"\n超限最严重的关节: {max_exceeded_joint} ({joint_names[max_exceeded_joint] if max_exceeded_joint < len(joint_names) else f'Joint_{max_exceeded_joint}'})")
        print(f"  超限次数: {max_exceeded_count}")
        print(f"  超限率: {(max_exceeded_count / len(df)) * 100:.2f}%")
        print(f"  最大力矩: {np.max(abs_torques[:, max_exceeded_joint]):.2f} Nm")
        print(f"  限制值: {torque_limits[max_exceeded_joint]:.1f} Nm")
        print(f"  超出限制: {np.max(abs_torques[:, max_exceeded_joint]) - torque_limits[max_exceeded_joint]:.2f} Nm")
    
    # 找出超限最严重的时刻
    print("\n超限最严重的时刻 (前10个):")
    print("-" * 80)
    print(f"{'行号':<8} {'时间(s)':<12} {'超限关节数':<12} {'最大超限值(Nm)':<15}")
    print("-" * 80)
    
    # 计算每行的超限情况
    exceeded_per_row = np.sum(exceeded, axis=1)
    max_exceed_per_row = np.max(abs_torques - limits[np.newaxis, :], axis=1)
    max_exceed_per_row[~np.any(exceeded, axis=1)] = 0  # 没有超限的行设为0
    
    # 获取最严重的10行
    top_indices = np.argsort(max_exceed_per_row)[::-1][:10]
    
    for idx in top_indices:
        if max_exceed_per_row[idx] > 0:
            row_num = idx + 2  # +2 because CSV has header and 0-indexed
            time_est = idx * dt  # 使用统一的dt
            print(f"{row_num:<8} {time_est:<12.3f} {exceeded_per_row[idx]:<12} {max_exceed_per_row[idx]:<15.2f}")
    
    # 按关节分组统计
    print("\n按关节类型统计:")
    print("-" * 80)
    leg_joints = list(range(12))  # 0-11
    waist_joint = [12]
    arm_joints = list(range(13, 27))  # 13-26
    
    leg_exceeded = np.sum(exceeded[:, leg_joints])
    waist_exceeded = np.sum(exceeded[:, waist_joint])
    arm_exceeded = np.sum(exceeded[:, arm_joints])
    
    print(f"腿部关节 (0-11): 超限 {leg_exceeded} 次")
    print(f"腰部关节 (12): 超限 {waist_exceeded} 次")
    print(f"手臂关节 (13-26): 超限 {arm_exceeded} 次")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    csv_file = "/home/dobot/Desktop/rl_deploy-develop-asap1/policy/atom/motor_20251121190310.csv"
    
    if not Path(csv_file).exists():
        print(f"错误: 文件不存在: {csv_file}")
        sys.exit(1)
    
    analyze_torque_limits(csv_file, torque_limits)

