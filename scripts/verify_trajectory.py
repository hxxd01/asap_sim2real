#!/usr/bin/env python3
"""
验证轨迹数据格式是否正确
"""

import pickle
import numpy as np
import sys
from pathlib import Path


def verify_trajectory(pkl_file):
    """验证 pickle 轨迹文件格式"""
    print(f"📖 验证文件: {pkl_file}\n")
    
    # 加载数据
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict):
        print("❌ 错误: 顶层数据应该是字典")
        return False
    
    # 检查是否有 motion0, motion1, ...
    motion_keys = [k for k in data.keys() if k.startswith('motion')]
    if not motion_keys:
        print("❌ 错误: 未找到 motion0, motion1, ... 键")
        return False
    
    print(f"✅ Episode数量: {len(motion_keys)}\n")
    
    all_valid = True
    required_keys = ['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'action',
                     'terminate', 'root_lin_vel', 'root_ang_vel', 'dof_vel', 
                     'motion_times', 'fps']
    
    for i, key in enumerate(sorted(motion_keys)):
        ep = data[key]
        print(f"=== {key} ===")
        
        # 检查必需字段
        missing = [k for k in required_keys if k not in ep]
        if missing:
            print(f"❌ 缺失字段: {missing}")
            all_valid = False
            continue
        
        # 检查数据类型和形状
        T = len(ep['motion_times'])
        print(f"  帧数 T: {T}")
        print(f"  时长: {ep['motion_times'][-1]:.2f}s")
        print(f"  FPS: {ep['fps']}")
        
        checks = [
            ('root_trans_offset', (T, 3), np.float32),
            ('pose_aa', (T, 35, 3), np.float32),
            ('dof', (T, 27), np.float32),
            ('root_rot', (T, 4), np.float32),
            ('action', (T, 27), np.float32),
            ('terminate', (T,), np.int64),
            ('root_lin_vel', (T, 3), np.float32),
            ('root_ang_vel', (T, 3), np.float32),
            ('dof_vel', (T, 27), np.float32),
            ('motion_times', (T,), np.float32),
        ]
        
        for field, expected_shape, expected_dtype in checks:
            arr = ep[field]
            if not isinstance(arr, np.ndarray):
                print(f"  ❌ {field}: 不是 numpy array")
                all_valid = False
                continue
            
            if arr.shape != expected_shape:
                print(f"  ❌ {field}: shape={arr.shape}, 期望={expected_shape}")
                all_valid = False
            elif arr.dtype != expected_dtype:
                print(f"  ❌ {field}: dtype={arr.dtype}, 期望={expected_dtype}")
                all_valid = False
            else:
                print(f"  ✅ {field}: {arr.shape}, {arr.dtype}")
        
        # 检查terminate标志
        term_count = np.sum(ep['terminate'])
        print(f"  终止标志数量: {term_count}")
        if term_count == 0:
            print(f"  ⚠️  警告: 没有terminate标志")
        elif term_count > 1:
            print(f"  ⚠️  警告: 有多个terminate标志")
        
        print()
    
    if all_valid:
        print("✅ 所有检查通过！数据格式正确")
        print(f"\n📊 统计:")
        total_frames = sum(len(data[k]['motion_times']) for k in motion_keys)
        total_duration = sum(data[k]['motion_times'][-1] for k in motion_keys if len(data[k]['motion_times']) > 0)
        print(f"  总帧数: {total_frames}")
        print(f"  总时长: {total_duration:.2f}s")
        print(f"  平均episode长度: {total_frames / len(motion_keys):.1f} 帧")
    else:
        print("❌ 存在错误，请检查")
    
    return all_valid


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python verify_trajectory.py <pickle_file>")
        print("示例: python verify_trajectory.py trajectory_20251204.pkl")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    if not Path(pkl_file).exists():
        print(f"❌ 文件不存在: {pkl_file}")
        sys.exit(1)
    
    verify_trajectory(pkl_file)



