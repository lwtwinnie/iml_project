import numpy as np
import pandas as pd

# 设置随机种子以确保结果可复现
np.random.seed(42)
#生成单侧数据集的大小
size = 2000
# 生成两个多元高斯分布，每个分布包含1000个点
mean1 = [500, 500]
cov1 = [[100000, 0], [0, 100000]]
points1 = np.random.multivariate_normal(mean1, cov1, size)

mean2 = [-230, -230]
cov2 = [[100000, 0], [0, 10000]]
points2 = np.random.multivariate_normal(mean2, cov2, size)

# 创建DataFrame保存生成的数据
df1 = pd.DataFrame(points1, columns=['x', 'y'])
df2 = pd.DataFrame(points2, columns=['x', 'y'])

# 添加 'category' 列，并赋值为 -1 和 1
df1['category'] = -1
df2['category'] = 1

# 引入噪声比例，修改部分点的 'category' 并添加 'noise' 列
a = 5
b = 5

# 对 p_category=1 的点进行修改
# 随机创建掩码
mask1 = np.random.choice([True, False], size=size, p=[a/100, 1-a/100])
#由 -1 更改为 1
df1.loc[mask1, 'category'] = 1
# 记录被修改的点，认为是噪声点为1，否则为0
df1['noise'] = np.where(mask1, 1, 0)

# 对 p_category=-1 的点进行修改
mask2 = np.random.choice([True, False], size=size, p=[b/100, 1-b/100])
#由 1 更改为 -1
df2.loc[mask2, 'category'] = -1
df2['noise'] = np.where(mask2, 1, 0)

# 合并两个DataFrame
dataset = pd.concat([df1, df2], ignore_index=True)

# 保存为 CSV 文件
dataset.to_csv('overlapping_5%_noise.csv', index=False)


import matplotlib.pyplot as plt

# 根据条件选择数据并绘制散点图
plt.figure(figsize=(10, 8))

# noise=1, category=-1 涂紫色
plt.scatter(dataset[(dataset['noise'] == 1) & (dataset['category'] == -1)]['x'],
            dataset[(dataset['noise'] == 1) & (dataset['category'] == -1)]['y'],
            c='purple', label='Noise, Category=-1')
# noise=0, category=-1 涂红色
plt.scatter(dataset[(dataset['noise'] == 0) & (dataset['category'] == -1)]['x'],
            dataset[(dataset['noise'] == 0) & (dataset['category'] == -1)]['y'],
            c='red', marker='x', label='not noise, Category=-1')

# noise=1, category=1 涂橙色
plt.scatter(dataset[(dataset['noise'] == 1) & (dataset['category'] == 1)]['x'],
            dataset[(dataset['noise'] == 1) & (dataset['category'] == 1)]['y'],
            c='orange', label='Noise, Category=1')

# noise=0, category=1 涂黄色
plt.scatter(dataset[(dataset['noise'] == 0) & (dataset['category'] == 1)]['x'],
            dataset[(dataset['noise'] == 0) & (dataset['category'] == 1)]['y'],
            c='yellow', marker='x', label='not noise, Category=1')



# 设置图例和标签
plt.legend()
plt.title('Visualization of Dataset')
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.show()
