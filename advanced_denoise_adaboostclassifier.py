import numpy as np
import pandas as pd
from numpy import sqrt, e, log
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.font_manager
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import csv
from scipy.spatial.distance import euclidean
from scipy.special import expit

class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.best_err = 1  # 最小的加权错误率
        self.best_fea_id = 0  # 最优特征id
        self.best_thres = 0  # 选定特征的最优阈值
        self.best_op = 1  # 阈值符号，其中 1: >, 0: <

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        n = X.shape[1]
        for i in range(n):
            feature = X[:, i]  # 选定特征列
            fea_unique = np.sort(np.unique(feature))  # 将所有特征值从小到大排序
            for j in range(len(fea_unique) - 1):
                thres = (fea_unique[j] + fea_unique[j + 1]) / 2  # 逐一设定可能阈值
                for op in (0, 1):
                    y_ = 2 * (feature >= thres) - 1 if op == 1 else 2 * (feature < thres) - 1  # 判断何种符号为最优
                    err = np.sum((y_ != y) * sample_weight)
                    if err < self.best_err:  # 当前参数组合可以获得更低错误率，更新最优参数
                        self.best_err = err
                        self.best_op = op
                        self.best_fea_id = i
                        self.best_thres = thres
        return self

    def predict(self, X):
        feature = X[:, self.best_fea_id]
        return 2 * (feature >= self.best_thres) - 1 if self.best_op == 1 else 2 * (feature < self.best_thres) - 1

    def score(self, X, y, sample_weight=None):
        y_pre = self.predict(X)
        if sample_weight is not None:
            return np.sum((y_pre == y) * sample_weight)
        return np.mean(y_pre == y)


class _AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
        self.k = 10  # k neighbors
        # 得到pca的维度
        self.d = 2  # dimension
        self.PCA = np.empty((0,) * self.d)
        self.betas = []
        self.deviation = []
        self.distances = np.empty(2000 * 2000)

    # —————————自己加的函数，只得到一次pca之间的距离即可，平常不调用——————
    def get_distances_and_store(self, X_pca):
        # 计算二维数组与其他点距离
        X_pca = np.array(X_pca)

        # Calculate pairwise distances using Euclidean distance
        distances = cdist(X_pca, X_pca, 'euclidean')

        with open(file_path, 'a') as file:
            for row in distances:
                file.write(','.join(map(str, row)) + '\n')

        return distances
    #读取distances_txt文件中的i行j列
    def get_distances_txt_ij_from_store(self,i,j):
        # 读取distances文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            lines = file.readlines()

        # 将文本内容解析为二维数组
        distances_matrix = [list(map(float, line.strip().split(','))) for line in lines]
        # 找到第i行第j列的数字（假设i和j从0开始）
        value = distances_matrix[i][j]
        # print(f"The value at row {i} and column {j} is: {value}")
        return value

    # 读取distances_txt文件中的第i行
    def get_distances_txt_row_i_from_store(self, i):
        # 读取distances文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            lines = file.readlines()

        # 将文本内容解析为二维数组
        distances_matrix = [list(map(float, line.strip().split(','))) for line in lines]
        # 找到第i行的内容（假设i从0开始）
        row_content = distances_matrix[i]
        # print(f"The values in row {i} are: {row_content}")
        return row_content
    def get_distances_txt_from_store(self):
        # 读取distances文件
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            lines = file.readlines()
        # 将文本内容解析为二维数组
        distances_matrix = [list(map(float, line.strip().split(','))) for line in lines]
        return distances_matrix



    def cal_similarity(self, X_pca, y):#返回每个点的similarity
        k = self.k  # 设置K值为最近k个邻居
        similarities = []
        for i in range(len(y)):
            # 1. 找到第i行除了第i个以外的最小k的索引
            indices = np.argsort(self.distances[i])[1:k + 1]  # 排除自身，选取最近的k个邻居

            # 2. 获取这些点的标签
            neighbor_labels = y[indices]

            # 3. 计算相同标签的比例
            similarity = np.mean(neighbor_labels == y[i])
            similarities.append(similarity)

        return similarities

    def cal_preference(self, X_pca, y):#返回每个点的preference
        k = self.k  # 设置K值为最近k个邻居
        preferences = []
        for i in range(len(y)):
            # 1. 找到第i行除了第i个以外的从小到大的索引
            indices = np.argsort(self.distances[i])

            # 2. 获取这些点的标签
            neighbors_labels = y[indices]

            # 3. 如果标签为和第i个点相同，放入相同标签的群1；如果标签为和第i个点不同，放入不同标签的群2。当一个群有k个的时候，停止添加这个群。当两个群都有k个的时候，停止。
            same_label_group_index=[]
            diff_label_group_index=[]
            for k in indices:
                if neighbors_labels[k] == y[i]:
                    same_label_group_index.append(k)
                else:
                    diff_label_group_index.append(k)
            same_label_group_index = same_label_group_index[1:k+1]
            diff_label_group_index = diff_label_group_index[:k]

            # 4. 得到1和2分别的中心点A和B
            center_A = np.mean(X_pca[same_label_group_index[:k]], axis=0)
            center_B = np.mean(X_pca[diff_label_group_index[:k]], axis=0)

            # 5. 计算i这个点到A的距离 # 6. 计算i这个点到B的距离
            dist_to_A = np.linalg.norm(X_pca[i] - center_A)
            dist_to_B = np.linalg.norm(X_pca[i] - center_B)

            # 7. 计算1中的点到A的平均距离 # 8. 计算2中的点到B的平均距离
            distances_to_center_A = [euclidean(point, center_A) for point in X_pca[same_label_group_index[:k]]]
            avg_dist_A = np.mean(distances_to_center_A)

            distances_to_center_B = [euclidean(point, center_A) for point in X_pca[diff_label_group_index[:k]]]
            avg_dist_B = np.mean(distances_to_center_B)

            # 9. 计算5/7  # 10. 计算6/8
            ratio_1 = dist_to_A / (avg_dist_A + 1e-9)
            ratio_2 = dist_to_B / (avg_dist_B+ 1e-9)

            # 11. 计算10/9
            preference = ratio_2/ratio_1
            preferences.append(preference)

        return preferences

    def store_betas(self, beta):
        # 存一下beta这个列表
        beta = np.array(beta)
        with open('DATA/BRFSS_result/beta.txt', 'w') as file:
            file.write(','.join(map(str, beta)) + '\n')

    def cal_betas(self, similarity, preference):
        betas = []
        for i in range(len(similarity)):
            balance = 1
            betas.append(similarity[i] * expit(-balance * preference[i]))
        betas = np.array(betas)
        self.betas = betas
        #print(betas)
        return

    def fit(self, X, y):
        # 计算每个点的pca降到d维度的值
        pca = PCA(n_components=self.d)
        X_pca = pca.fit_transform(X)
        self.PCA = X_pca
        #获得self.distances
        self.distances = self.get_distances_and_store(X_pca)#重新生成
        # 可以反复调用的工具函数：
        # print(self.get_distances_txt_ij_from_store(0,2))#得到点i与点j的距离
        # print(self.get_distances_txt_row_i_from_store(0))#得到点i与其他点的距离
        # print(self.get_distances_txt_from_store())


        sample_weight = np.ones(len(X)) / len(X)  # 初始化样本权重为 1/N
        for i in range(self.n_estimators):
            dtc = DecisionTreeClassifierWithWeight().fit(X, y, sample_weight)  # 训练弱学习器
            print('iteration:',i)
            alpha = 1 / 2 * np.log((1 - dtc.best_err) / dtc.best_err)  # 权重系数
            y_pred = dtc.predict(X)
            sample_weight *= np.exp(-alpha * y_pred * y)  # 更新迭代样本权重
            sample_weight /= np.sum(sample_weight)  # 样本权重归一化


            self.estimators.append(dtc)
            self.alphas.append(alpha)

            if i >= 30:
            #截至目前已有的所有分类器的综合预测结果：
                y_pred_sum = self.temple_predict(i, X)
                # y_pred_sum=y_pred
                similarity = self.cal_similarity(self.PCA, y_pred_sum)  #返回每个点的similarity
                preference = self.cal_preference(self.PCA, y_pred_sum)  #返回每个点的preference
                self.cal_betas(similarity, preference)  #得到每个点的beta（置信度）存入self.beta
                sample_weight *= np.exp(self.betas)  # 更新迭代样本权重
                sample_weight /= np.sum(sample_weight)  # 样本权重归一化

        return self

    def temple_predict(self, iterations, X):
        y_pred = np.empty((len(X), iterations))  # 预测结果二维数组，其中每一列代表一个弱学习器的预测结果
        for i in range(iterations):
            y_pred[:, i] = self.estimators[i].predict(X)
        alphas = self.alphas[: iterations]
        y_pred = y_pred * np.array(alphas)  # 将预测结果与训练权重乘积作为集成预测结果
        return 2 * (np.sum(y_pred, axis=1) > 0) - 1  # 以0为阈值，判断并映射为-1和1

    def predict(self, X):
        y_pred = np.empty((len(X), self.n_estimators))  # 预测结果二维数组，其中每一列代表一个弱学习器的预测结果
        for i in range(self.n_estimators):
            y_pred[:, i] = self.estimators[i].predict(X)
        y_pred = y_pred * np.array(self.alphas)  # 将预测结果与训练权重乘积作为集成预测结果
        return 2 * (np.sum(y_pred, axis=1) > 0) - 1  # 以0为阈值，判断并映射为-1和1

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        print('accuracy')
        return accuracy

test_manmade_set=0

# -----------------------------test----------------------------
file_path = 'DATA\MAN_MADE_result\distances.txt'

with open(file_path, 'w') as wins_file:
    wins_file.write('')


df = pd.read_csv('DATA\MAN_MADE\overlapping_5%_noise.csv')
#预处理：转为 int 类型
df['category'] = df['category'].apply(lambda x: int(x))
#预处理：去掉noise标识
df = df.drop(columns=['noise'])

#读取 X和 y
X = df.drop(columns=['category'])
X = X.to_numpy()
y = df['category']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
print(_AdaBoostClassifier().fit(X_train, y_train).score(X_test, y_test))
