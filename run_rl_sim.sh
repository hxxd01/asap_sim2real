#!/bin/bash

# 设置临时的本地DDS配置
export CYCLONEDDS_URI="file://$(pwd)/config/cyclonedds_local.xml"

# 运行仿真程序
./build/rl_sim "$@"

# 运行后自动恢复环境（可选）
unset CYCLONEDDS_URI