import numpy as np
import pandas as pd

# 设置随机种子以确保结果可复现
np.random.seed(42)
#生成数据集的大小
size=1000
# 生成两个多元高斯分布，每个分布包含1000个点
mean1 = [500, 500]
cov1 = [[100000, 0], [0, 100000]]
points1 = np.random.multivariate_normal(mean1, cov1, size)

mean2 = [-500, -500]
cov2 = [[100000, 0], [0, 10000]]
points2 = np.random.multivariate_normal(mean2, cov2, size)

# 创建DataFrame保存生成的数据
df1 = pd.DataFrame(points1, columns=['x', 'y'])
df2 = pd.DataFrame(points2, columns=['x', 'y'])

# 添加 'category' 列，并赋值为 1 和 2
df1['category'] = 1
df2['category'] = 2

# 引入噪声，修改部分点的 'category' 并添加 'noise' 列
a = 5
b = 5

# 对 p_category=1 的点进行修改
mask1 = np.random.choice([True, False], size=size, p=[a/100, 1-a/100])
df1.loc[mask1, 'category'] = 2
df1['noise'] = np.where(mask1, 1, 0)

# 对 p_category=2 的点进行修改
mask2 = np.random.choice([True, False], size=size, p=[b/100, 1-b/100])
df2.loc[mask2, 'category'] = 1
df2['noise'] = np.where(mask2, 1, 0)

# 合并两个DataFrame
dataset = pd.concat([df1, df2], ignore_index=True)

# 保存为 CSV 文件
dataset.to_csv('dataset.csv', index=False)


import matplotlib.pyplot as plt

# 根据条件选择数据并绘制散点图
plt.figure(figsize=(10, 8))

# noise=1, category=1 涂红色
plt.scatter(dataset[(dataset['noise'] == 1) & (dataset['category'] == 1)]['x'],
            dataset[(dataset['noise'] == 1) & (dataset['category'] == 1)]['y'],
            c='red', label='Noise=1, Category=1')

# noise=1, category=2 涂蓝色
plt.scatter(dataset[(dataset['noise'] == 1) & (dataset['category'] == 2)]['x'],
            dataset[(dataset['noise'] == 1) & (dataset['category'] == 2)]['y'],
            c='blue', label='Noise=1, Category=2')

# noise=0, category=1 涂蓝色
plt.scatter(dataset[(dataset['noise'] == 0) & (dataset['category'] == 1)]['x'],
            dataset[(dataset['noise'] == 0) & (dataset['category'] == 1)]['y'],
            c='blue', marker='x', label='Noise=0, Category=1')

# noise=0, category=2 涂红色
plt.scatter(dataset[(dataset['noise'] == 0) & (dataset['category'] == 2)]['x'],
            dataset[(dataset['noise'] == 0) & (dataset['category'] == 2)]['y'],
            c='red', marker='x', label='Noise=0, Category=2')

# 设置图例和标签
plt.legend()
plt.title('Visualization of Dataset')
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.show()
