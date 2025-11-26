#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 27 DOF joint names for Atom robot
JOINT_NAMES = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow_pitch', 'left_elbow_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow_pitch', 'right_elbow_roll', 'right_wrist_pitch', 'right_wrist_yaw'
]

# Joint groups for better visualization
JOINT_GROUPS = {
    'Left Leg': list(range(0, 6)),
    'Right Leg': list(range(6, 12)),
    'Waist': [12],
    'Left Arm': list(range(13, 20)),
    'Right Arm': list(range(20, 27))
}


def get_joint_name(idx):
    """Get joint name from index"""
    if 0 <= idx < len(JOINT_NAMES):
        return JOINT_NAMES[idx]
    return f"joint_{idx}"




def plot_group(t, df, cols, title, filename, use_joint_names=True):
    """Plot a group of columns in a separate figure"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for col in cols:
        # Extract joint index from column name if applicable
        label = col
        if use_joint_names:
            for prefix in ['joint_pos_', 'joint_vel_', 'tau_est_', 'cmd_q_', 'cmd_tau_']:
                if col.startswith(prefix):
                    try:
                        idx = int(col.replace(prefix, ''))
                        label = f"{get_joint_name(idx)}"
                    except ValueError:
                        pass
                    break
        
        ax.plot(t, df[col], label=label, linewidth=1.2, alpha=0.85)
    
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f"{title}\n{filename}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if len(cols) <= 10:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    else:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    return fig


def plot_joint_group_subplots(t, df, prefix, group_name, joint_indices, filename):
    """Plot joints from a specific group in subplots"""
    n_joints = len(joint_indices)
    if n_joints == 0:
        return None
    
    # Calculate subplot layout
    n_cols = min(3, n_joints)
    n_rows = (n_joints + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_joints == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for i, joint_idx in enumerate(joint_indices):
        col_name = f"{prefix}{joint_idx}"
        if col_name not in df.columns:
            continue
        
        ax = axes[i]
        ax.plot(t, df[col_name], linewidth=1.5, color=f'C{i}', alpha=0.9)
        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel(prefix.replace('_', ' ').title(), fontsize=10)
        ax.set_title(get_joint_name(joint_idx), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide unused subplots
    for i in range(n_joints, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f"{group_name} - {prefix.replace('_', ' ').title()}\n{filename}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="可视化 RL_Real 输出的 motor_*.csv 或 estimation_real_*.csv"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="CSV 文件路径（例如 policy/atom/motor_20251121190310.csv）",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=500.0,
        help="采样频率 Hz（默认 500）",
    )
    parser.add_argument(
        "--cols",
        type=str,
        default="",
        help="自定义要绘制的列名，逗号分隔，例如: x,y,z,est_lin_x,est_lin_y",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="下采样步长，>1 可以减少点数，例如 10 表示每 10 个点取一个",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存图片而不是显示（保存为PNG格式）",
    )
    parser.add_argument(
        "--group-by-body",
        action="store_true",
        help="按身体部位分组显示（左腿、右腿、左臂、右臂等），每个关节单独子图",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="开始时间（秒），例如 --start 2.0 从2秒开始",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="结束时间（秒），例如 --end 10.0 到10秒结束（不指定则到数据末尾）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="持续时间（秒），例如 --duration 5.0 显示5秒数据（优先级高于--end）",
    )
    args = parser.parse_args()

    # 读 CSV
    df = pd.read_csv(args.csv_path)
    df = df.dropna(axis=1, how="all")

    print("=" * 80)
    print(f"CSV文件: {args.csv_path}")
    print(f"原始数据行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    
    # 生成时间轴
    dt = 1.0 / args.freq
    total_duration = len(df) * dt
    print(f"总时长: {total_duration:.2f} 秒")
    
    # 时间范围裁剪
    start_time = max(0.0, args.start)
    if args.duration is not None:
        end_time = start_time + args.duration
    elif args.end is not None:
        end_time = args.end
    else:
        end_time = total_duration
    
    # 转换为索引
    start_idx = int(start_time * args.freq)
    end_idx = int(end_time * args.freq)
    
    # 边界检查
    start_idx = max(0, min(start_idx, len(df)))
    end_idx = max(start_idx, min(end_idx, len(df)))
    
    if start_idx > 0 or end_idx < len(df):
        print(f"时间范围裁剪: {start_time:.2f}s 到 {end_time:.2f}s")
        print(f"裁剪后数据行数: {end_idx - start_idx} (索引 {start_idx} 到 {end_idx})")
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    print("=" * 80)
    print("列名如下：")
    print(list(df.columns))
    print("=" * 80)

    # 生成时间轴（从start_time开始）
    t = (df.index.to_numpy() * dt) + start_time

    # 下采样
    if args.downsample > 1:
        df = df.iloc[::args.downsample, :].reset_index(drop=True)
        t = t[::args.downsample]
        print(f"下采样: 每 {args.downsample} 个点取一个，剩余 {len(df)} 个点")
        print("=" * 80)

    filename = Path(args.csv_path).name
    figures = []

    # 如果用户指定了 --cols，就按自定义列画
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip() in df.columns]
        if not cols:
            raise ValueError("指定的列名在 CSV 中都不存在，请检查 --cols 参数。")
        
        fig = plot_group(t, df, cols, "Custom Selection", filename, use_joint_names=True)
        figures.append(("custom", fig))
    else:
        # 自动分组可视化
        base_groups = [
            (["x", "y", "z"], "Base Position [m]"),
            (["r", "p", "yaw"], "Base RPY [rad]"),
            (["est_lin_x", "est_lin_y", "est_lin_z"], "Estimated Linear Velocity [m/s]"),
            (["network_lin_x", "network_lin_y", "network_lin_z"], "Network Linear Output [m/s]"),
            (["ang_x", "ang_y", "ang_z"], "Angular Velocity [rad/s]"),
            (["est_contact_force_left", "est_contact_force_right"], "Contact Force Estimation"),
            (["motion_phase"], "Motion Phase"),
        ]

        prefix_groups = [
            ("joint_pos_", "Joint Positions [rad]"),
            ("joint_vel_", "Joint Velocities [rad/s]"),
            ("tau_est_", "Estimated Joint Torque [Nm]"),
            ("cmd_q_", "Commanded Joint Positions [rad]"),
            ("cmd_tau_", "Commanded Joint Torque [Nm]"),
        ]

        # Plot base groups (non-joint data)
        for cols, title in base_groups:
            existing = [c for c in cols if c in df.columns]
            if existing:
                fig = plot_group(t, df, existing, title, filename, use_joint_names=False)
                figures.append((title, fig))

        # Plot joint-related data
        if args.group_by_body:
            # Group by body parts with subplots
            for prefix, title in prefix_groups:
                for group_name, joint_indices in JOINT_GROUPS.items():
                    fig = plot_joint_group_subplots(t, df, prefix, group_name, joint_indices, filename)
                    if fig:
                        figures.append((f"{title} - {group_name}", fig))
        else:
            # Plot all joints together
            for prefix, title in prefix_groups:
                matched = [c for c in df.columns if c.startswith(prefix)]
                # Sort by joint index
                matched = sorted(matched, key=lambda x: int(x.replace(prefix, '')) if x.replace(prefix, '').isdigit() else 999)
                if matched:
                    fig = plot_group(t, df, matched, title, filename, use_joint_names=True)
                    figures.append((title, fig))

    if not figures:
        raise RuntimeError("没有找到可以绘制的数据！")

    print(f"\n总共生成 {len(figures)} 个图表")
    print("=" * 80)

    # Save or show figures
    if args.save:
        output_dir = Path(args.csv_path).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        base_name = Path(args.csv_path).stem
        
        for title, fig in figures:
            # Clean title for filename
            safe_title = title.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
            output_path = output_dir / f"{base_name}_{safe_title}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"保存: {output_path}")
        
        print(f"\n所有图表已保存到: {output_dir}")
    else:
        print("\n显示图表... (关闭所有窗口以退出)")
        plt.show()


if __name__ == "__main__":
    main()
