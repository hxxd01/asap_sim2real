#!/usr/bin/env python3
import pandas as pd
import numpy as np

csv_file = "/home/dobot/Desktop/rl_deploy-develop-asap1/policy/atom/motor_20251120191132.csv"
df = pd.read_csv(csv_file)

print("CSV shape:", df.shape)
print("\n列名:")
print(df.columns.tolist())

print("\n检查cmd_tau列:")
cmd_tau_cols = [f'cmd_tau_{i}' for i in range(27)]
print(f"需要的列: {cmd_tau_cols[:5]}...{cmd_tau_cols[-2:]}")

# 检查列是否存在
for col in cmd_tau_cols:
    if col in df.columns:
        print(f"✓ {col} 存在")
    else:
        print(f"✗ {col} 不存在")

print("\n前5行的cmd_tau_20数据:")
print(df['cmd_tau_20'].head())

print("\ncmd_tau_20统计:")
print(df['cmd_tau_20'].describe())

print("\ncmd_tau_20绝对值最大值:")
print(np.abs(df['cmd_tau_20']).max())

print("\n前3行所有cmd_tau列:")
print(df[cmd_tau_cols].head(3))

