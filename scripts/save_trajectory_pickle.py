#!/usr/bin/env python3
"""
将 C++ 保存的 JSON 轨迹文件转换为 pickle 格式
与 Python 采集脚本的输出格式完全一致
"""

import json
import numpy as np
import joblib
import sys
from pathlib import Path


def json_to_pickle(json_file):
    """
    转换 JSON 轨迹文件为 pickle 格式
    Args:
        json_file: JSON 文件路径
    """
    print(f"📖 读取 JSON 文件: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 转换数据格式
    result = {}
    episode_count = data.get('episode_count', 1)
    
    print(f"📊 总episode数: {episode_count}")
    
    for i in range(episode_count):
        episode_key = f'motion{i}'
        if episode_key not in data:
            print(f"⚠️  警告: 未找到 {episode_key}")
            continue
        
        ep_data = data[episode_key]
        
        # 转换每个字段为 numpy array
        converted_ep = {}
        
        # root_trans_offset: [T, 3]
        converted_ep['root_trans_offset'] = np.array(ep_data['root_trans_offset'], dtype=np.float32)
        
        # pose_aa: [T, 35, 3]
        converted_ep['pose_aa'] = np.array(ep_data['pose_aa'], dtype=np.float32)
        
        # dof: [T, 27]
        converted_ep['dof'] = np.array(ep_data['dof'], dtype=np.float32)
        
        # root_rot: [T, 4]
        converted_ep['root_rot'] = np.array(ep_data['root_rot'], dtype=np.float32)
        
        # action: [T, 27]
        converted_ep['action'] = np.array(ep_data['action'], dtype=np.float32)
        
        # terminate: [T]
        converted_ep['terminate'] = np.array(ep_data['terminate'], dtype=np.int64)
        
        # root_lin_vel: [T, 3]
        converted_ep['root_lin_vel'] = np.array(ep_data['root_lin_vel'], dtype=np.float32)
        
        # root_ang_vel: [T, 3]
        converted_ep['root_ang_vel'] = np.array(ep_data['root_ang_vel'], dtype=np.float32)
        
        # dof_vel: [T, 27]
        converted_ep['dof_vel'] = np.array(ep_data['dof_vel'], dtype=np.float32)
        
        # motion_times: [T]
        converted_ep['motion_times'] = np.array(ep_data['motion_times'], dtype=np.float32)
        
        # fps
        converted_ep['fps'] = float(ep_data.get('fps', 50.0))
        
        result[episode_key] = converted_ep
        
        T = len(converted_ep['motion_times'])
        duration = converted_ep['motion_times'][-1] if T > 0 else 0.0
        print(f"  {episode_key}: {T} 帧, {duration:.2f}秒")
    
    # 使用 joblib 保存（兼容不同 numpy 版本）
    output_file = json_file.replace('.json', '.pkl')
    joblib.dump(result, output_file, compress=0)
    
    print(f"\n✅ 已保存为: {output_file}")
    print(f"📦 格式: {{'motion0': {...}, 'motion1': {...}, ...}}")
    print(f"🎯 使用 joblib 保存，兼容训练环境")
    
    return output_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python save_trajectory_pickle.py <json_file>")
        print("示例: python save_trajectory_pickle.py trajectory_20251204.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not Path(json_file).exists():
        print(f"❌ 文件不存在: {json_file}")
        sys.exit(1)
    
    json_to_pickle(json_file)

