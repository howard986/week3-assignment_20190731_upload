# In[1]: Logistic Regression
import numpy as np

# In[2]: 
class LogisticRegression():
    # 初始化函数
    def __init__(self):
        self.l_rate = 0.01 
        self.w = None

    # sigmoid激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 预测函数
    # X是形如[m, n]的矩阵: m是样本数目,n是特征数目+1(因为把bias也统一到X中)
    # w是[n, 1]的向量
    # 返回的是[m, 1]的向量
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.w))

    # 计算梯度
    # y是形如[m, 1]的向量
    # 返回的是形如[n, 1]的向量,是每个参数的梯度
    def gradient(self, X, y):
        m = len(y)
        y_pre = self.predict(X)
        return np.dot(X.T, y_pre - y) / m 

    # 更新w参数
    # w是[n, 1]的列向量
    def update_para(self, X, y):
        self.w -= self.l_rate * self.gradient(X, y)

    # 损失函数 返回一个标量
    def loss(self, X, y):
        m = len(y)
        h = self.predict(X)
        y_pre = np.zeros(y.shape)
        y_pre[h>0.5] = 1
        accuracy = sum(y_pre==y) / len(y_pre)
        return accuracy, np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m

    def accuracy(self, X, y):
        pass

    # 训练数据
    def train(self, X, y, l_rate=0.01, max_itr=1000, batch_size=50):
        # 添加bias
        X = np.array(X).reshape(len(X), -1)
        X = np.hstack((X, np.ones((X.shape[0],1))))
        y = np.array(y).reshape(len(X), -1)

        self.w = np.zeros((X.shape[1], 1))
        self.l_rate = l_rate
        for i in range(max_itr):
            batch_idxes = list(np.random.choice(X.shape[0], batch_size))
            batch_X = [X[i] for i in batch_idxes]
            batch_y = [y[i] for i in batch_idxes]
            self.update_para(X, y)
            #print('w:{0}'.format(self.w))
            accu, loss = self.loss(X, y)
            print('loss is {0}, accuracy is {1}'.format(loss, accu))

        return self.w

    # 计算测试数据的误差
    def test(self, X, y):
        X = np.array(X).reshape(len(X), -1)
        X = np.hstack((X, np.ones((X.shape[0],1))))
        y = np.array(y).reshape(len(X), -1)
        return self.loss(X, y)

