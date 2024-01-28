import matplotlib.pyplot as plt

# 从文本中读取数据
with open('D:/pycharm/PycharmProjects/pythonProject_tf1/cs182/DATA/BRFSS_result/beta.txt', 'r') as file:
    data = file.readline().strip().split(',')

# 将字符串列表转换为浮点数列表
data = [float(value) for value in data]

# 绘制频率图
plt.hist(data, bins=50, edgecolor='black')  # 你可以调整 bins 的数量
plt.title('Frequency Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
