#!/bin/bash

# 检查是否安装了cset工具
if ! command -v cset &> /dev/null; then
    echo "错误：未安装cpuset工具，请先安装（例如：sudo apt install cpuset）"
    exit 1
fi

# 绑定 CPU 0-5，并允许内核线程运行在所有 CPU 上
echo "创建CPU屏蔽集（0-5）..."
if ! cset shield --cpu=0-5 --kthread=on; then
    echo "错误：创建CPU屏蔽集失败"
    exit 1
fi

# 启动程序
echo "启动rl_real_atom程序..."
nohup cset shield --exec -- bash -c "./build/rl_real_atom" > real_start.log 2>&1 &
start_pid=$!

# 等待进程启动（最多等待10秒）
wait_count=0
max_wait=10
echo "等待程序启动..."
while ! pgrep -x "rl_real_atom" &> /dev/null; do
    sleep 1
    wait_count=$((wait_count + 1))
    if [ $wait_count -ge $max_wait ]; then
        echo "错误：rl_real_atom启动超时"
        exit 1
    fi
done

# 循环确保所有线程绑定到 CPU 0-5，限制最大重试次数
max_retries=10
retry_count=0
echo "开始检查并绑定线程..."

while [ $retry_count -lt $max_retries ]; do
    echo "第 $((retry_count + 1)) 次检查绑定情况："
    ps -eLo pid,tid,comm,psr | grep -v grep | grep rl_real_atom

    # 检查是否仍有线程跑在 CPU 6、7 上
    if ps -eLo pid,tid,comm,psr | grep -v grep | grep rl_real_atom | awk '$4 > 5 {exit 1}'; then
        echo "所有线程都成功绑定在 CPU 0-5！"
        break
    else
        echo "仍有线程未正确绑定，重新尝试..."

        # 重新绑定所有线程
        for pid in $(pgrep -x rl_real_atom); do
            # 只处理数字形式的线程ID
            for tid in $(ls -1 /proc/$pid/task 2>/dev/null | grep -E '^[0-9]+$'); do
                if ! sudo taskset -p -c 0-5 $tid &> /dev/null; then
                    echo "警告：无法绑定线程 $tid 到CPU 0-5"
                fi
            done
        done

        retry_count=$((retry_count + 1))
        if [ $retry_count -eq $max_retries ]; then
            echo "警告：达到最大重试次数($max_retries)，部分线程可能未正确绑定"
            break
        fi
        sleep 1
    fi
done

echo "脚本执行完成"
