# In[1]: Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import random
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as sklearn_lgsr
def gen_data():
    w = np.random.randint(0, 10) + np.random.rand()
    b = np.random.randint(0, 5) + np.random.rand()
    num_sample = 100
    X = (np.random.randint(0, 100, size=100) * np.random.rand()).reshape(1, -1)
    y = (X * w + b + np.random.rand() * np.random.randint(-1, 2)).reshape(1, -1)
    return X, y, w, b

# In[6]:
def LinearRegressionTest():
    X, y, w_org, b_org = gen_data()

    learning_rate = 0.1
    max_itr = 500
    # 生成测试集
    m = X.shape[1]
    X = X[0].tolist()
    y = y[0].tolist()
    m = len(X)
    index = list(range(100))
    np.random.shuffle(index) # 将index乱序
    train_idxes = index[0:70]
    train_X = [X[i] for i in train_idxes]
    train_y = [y[i] for i in train_idxes]
    mean = np.mean(train_X)
    std = np.std(train_X)
    train_X = (train_X - mean) / std
    # 生成训练集
    test_idxes = index[70:100]
    test_X = [X[i] for i in test_idxes]
    test_y = [y[i] for i in test_idxes]
    test_X = (test_X - mean) / std
    # 训练
    lr = LinearRegression()
    w = lr.train(train_X, train_y, l_rate=learning_rate, max_itr=max_itr, batch_size=50)
    print('test loss is {0}'.format(lr.test(test_X, test_y)))
    x_ax = range(0, 100, 10)
    x_ax = (x_ax - mean) / std
    y_ax = w[0] * x_ax + w[1]

    plt.scatter(train_X, train_y, marker = 'o', c='b')
    plt.scatter(test_X, test_y, marker = 'o', c='r')
    plt.plot(x_ax, y_ax)
    plt.show()

def LogisticRegressionTest():
    # 导入乳腺癌数据集
    cancer_data = load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    lgsr = LogisticRegression()
    lgsr.train(X_train_std, y_train, 0.01, 1000)
    accu, loss = lgsr.test(X_test_std, y_test)
    print('自己手写的逻辑回归算法的结果：loss {0}, accuracy {1}'.format(loss, accu))

    clf = sklearn_lgsr()
    clf.fit(X_train_std, y_train)
    print('调用sklearn的逻辑回归模块的准确率是：', clf.score(X_test_std, y_test))

if __name__ == '__main__':
    LinearRegressionTest()
    LogisticRegressionTest()
