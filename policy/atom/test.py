import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# 假设你的CSV文件名为'data.csv'
df = pd.read_csv('policy/atom/estimation_20251022163649.csv')

# 设置图形大小
plt.figure(figsize=(14, 8))

# 绘制各维度的速度估计值与实际值
labels = ['Linear X', 'Linear Y', 'Linear Z']
for i, label in enumerate(labels):
    # 创建子图
    plt.subplot(3, 1, i + 1)  # 3行1列的布局
    plt.plot(df.index, df[f'est_lin_{label[-1].lower()}'], label=f'Estimated {label}')
    plt.plot(df.index, df[f'actual_lin_{label[-1].lower()}'], label=f'Actual {label}')
    plt.title(label)
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.legend()

# 调整布局，防止重叠
plt.tight_layout()
# 显示图像
plt.show()
