import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
# 跑的话需要pip install xgboost
# 读取数据
df = pd.read_csv('DATA\MAN_MADE\overlapping_5%_noise.csv')

# 预处理：转为 int 类型
df['category'] = df['category'].astype(int)

# 预处理：去掉 noise 标识
df = df.drop(columns=['noise'])

# 读取 X 和 y
X = df.drop(columns=['category'])
y = df['category']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

y_train = [0 if label == -1 else label for label in y_train]
y_test= [0 if label == -1 else label for label in y_test]

# 将数据转换为 XGBoost 的 DMatrix 格式
dtrain = xgb.core.DMatrix(X_train, label=y_train)
dtest = xgb.core.DMatrix(X_test, label=y_test)

# 定义参数
params = {
    'objective': 'binary:logistic',  # 二分类问题
    'eval_metric': 'error',  # 评估指标为错误率
    'eta': 0.05,  # 学习率
    'max_depth': 3  # 决策树最大深度
}

# 训练模型
num_round = 100  # 迭代次数
model = xgb.train(params, dtrain, num_round)

# 在测试集上进行预测
y_pred = model.predict(dtest)
y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")
