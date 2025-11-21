#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="可视化 RL_Real 输出的 estimation_real_*.csv"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="CSV 文件路径（例如 policy/atom/estimation_real_20251120143010.csv）",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=500.0,
        help="采样频率 Hz（默认 500，对应你 1000Hz 控制里每 2 次写一次）",
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
    args = parser.parse_args()

    # 读 CSV，顺便把全空列（比如多出来的那一列）去掉
    df = pd.read_csv(args.csv_path)
    df = df.dropna(axis=1, how="all")

    print("列名如下：")
    print(list(df.columns))

    # 生成时间轴
    dt = 1.0 / args.freq
    t = df.index.to_numpy() * dt

    # 下采样
    if args.downsample > 1:
        df = df.iloc[::args.downsample, :].reset_index(drop=True)
        t = t[::args.downsample]

    # 如果用户指定了 --cols，就按自定义列画
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip() in df.columns]
        if not cols:
            raise ValueError("指定的列名在 CSV 中都不存在，请检查 --cols 参数。")
        n = len(cols)
        plt.figure(figsize=(10, 2.5 * n))
        for i, col in enumerate(cols, start=1):
            plt.subplot(n, 1, i)
            plt.plot(t, df[col])
            plt.ylabel(col)
            if i == 1:
                plt.title(args.csv_path)
            if i == n:
                plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.show()
        return

    # 否则使用默认分组 / 自动分组可视化
    base_groups = [
        (["x", "y", "z"], "Base Position [m]"),
        (["r", "p", "yaw"], "Base RPY [rad]"),
        (["est_lin_x", "est_lin_y", "est_lin_z"], "Estimated Linear Velocity [m/s]"),
        (
            ["network_lin_x", "network_lin_y", "network_lin_z"],
            "Network Linear Output [m/s]",
        ),
        (["est_contact_force_left", "est_contact_force_right"], "Contact Force Estimation"),
    ]

    prefix_groups = [
        ("joint_pos_", "Joint Positions"),
        ("joint_vel_", "Joint Velocities"),
        ("tau_est_", "Estimated Joint Torque"),
        ("cmd_q_", "Commanded Joint Positions"),
        ("cmd_tau_", "Commanded Joint Torque"),
    ]

    groups_to_plot = []

    for cols, title in base_groups:
        existing = [c for c in cols if c in df.columns]
        if existing:
            groups_to_plot.append((existing, title))

    for prefix, title in prefix_groups:
        matched = [c for c in df.columns if c.startswith(prefix)]
        if matched:
            groups_to_plot.append((matched, title))

    if "motion_phase" in df.columns:
        groups_to_plot.append((["motion_phase"], "Motion Phase"))

    if not groups_to_plot:
        raise RuntimeError("默认分组里的列在 CSV 中都没找到，请用 --cols 手动指定要画的列。")

    n = len(groups_to_plot)
    plt.figure(figsize=(10, 3 * n))
    for i, (cols, title) in enumerate(groups_to_plot, start=1):
        plt.subplot(n, 1, i)
        for col in cols:
            plt.plot(t, df[col], label=col)
        plt.ylabel(title)
        plt.legend(loc="best")
        if i == 1:
            plt.title(args.csv_path)
        if i == n:
            plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
