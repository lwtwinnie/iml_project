import numpy as np
import pandas as pd

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成数据集的大小
size = 1000

# 生成三个多元高斯分布，每个分布包含1000个点
mean1 = [500, 500, 500]
cov1 = [[100000, 0, 0], [0, 100000, 0], [0, 0, 100000]]
points1 = np.random.multivariate_normal(mean1, cov1, size)

mean2 = [-500, -500, -500]
cov2 = [[100000, 0, 0], [0, 100000, 0], [0, 0, 100000]]
points2 = np.random.multivariate_normal(mean2, cov2, size)

# 创建DataFrame保存生成的数据
df1 = pd.DataFrame(points1, columns=['x', 'y', 'z'])
df2 = pd.DataFrame(points2, columns=['x', 'y', 'z'])

# 添加 'category' 列，并赋值为 1 和 2
df1['category'] = 0
df2['category'] = 1

# 引入噪声，修改部分点的 'category' 并添加 'noise' 列
a = 5
b = 5

# 对 p_category=1 的点进行修改
mask1 = np.random.choice([True, False], size=size, p=[a/100, 1-a/100])
df1.loc[mask1, 'category'] = 1
df1['noise'] = np.where(mask1, 1, 0)

# 对 p_category=2 的点进行修改
mask2 = np.random.choice([True, False], size=size, p=[b/100, 1-b/100])
df2.loc[mask2, 'category'] = 0
df2['noise'] = np.where(mask2, 1, 0)

# 合并两个DataFrame
dataset = pd.concat([df1, df2], ignore_index=True)

# 保存为 CSV 文件
dataset.to_csv('dataset_3d.csv', index=False)

# 可视化三维数据集
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 根据条件选择数据并绘制散点图
ax.scatter(dataset['x'], dataset['y'], dataset['z'], c=dataset['category'], cmap='viridis', s=10, alpha=0.5)

# 设置图例和标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of 3D Dataset')

# 显示图形
plt.show()
