# 导入必要的库
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
# 读取数据集
dataset = pd.read_csv('dataset.csv')

# 分割数据集为特征和标签
X = dataset[['x', 'y']]
y = dataset['category']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用SAMME.R算法构建AdaBoost分类器
base_classifier = DecisionTreeClassifier(max_depth=1)  # 基分类器选择决策树，您可以根据需要调整参数
adaboost_classifier = AdaBoostClassifier(base_classifier, algorithm='SAMME.R', n_estimators=50)  # 使用SAMME.R算法

# 训练分类器
adaboost_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = adaboost_classifier.predict(X_test)

# 评估分类器性能
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
