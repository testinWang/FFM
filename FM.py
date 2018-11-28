# coding=utf-8
__author__ = 'Wanghailong'

import numpy as np
np.random.seed(0)
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    sigmoid函数对z求一阶偏导
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """
        平方误差损失函数
        :param a: 预测值
        :param y: 真实值
        :return:
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        损失函数对z求偏导
        :param z: x的线性函数
        :param a:
        :param y:
        :return:
        """
        return (a - y) * sigmoid_prime(z)


class FM(object):
    def __init__(self, train, valid, k, eta, maxecho, r2, cost=QuadraticCost):
        """
        构造函数
        :param train: 训练数据
        :param valid: 验证数据
        :param k: 矩阵V的第2维
        :param eta: 固定学习率
        :param maxecho: 最多迭代次数
        :param r2: R2小于该值后可停止迭代
        :param cost: 损失函数
        """
        self.train_x = train[:, :-1]
        self.train_y = train[:, -1:]
        self.valid_x = valid[:, :-1]
        self.valid_y = valid[:, -1:]
        self.var_y = np.var(self.valid_y)  # y的方差，在每轮迭代后计算R2时要用到
        self.k = k
        self.eta = float(eta)
        self.maxecho = maxecho
        self.r2 = r2
        self.cost = cost
        # 用正态分布随机初始化参数W和V
        self.w0 = np.random.randn()
        self.w = np.random.randn(1, self.train_x.shape[1])
        self.v = np.random.randn(self.train_x.shape[1], self.k)

    def shuffle_data(self):
        """
        每轮训练之前都随机打乱样本顺序
        :return:
        """
        ids = range(len(self.train_x))
        random.shuffle(ids)
        self.train_x = self.train_x[ids]
        self.train_y = self.train_y[ids]

    def predict(self, x):
        """
        根据x求y
        :param x:
        :return:
        """
        z = self.w0 + np.dot(self.w, x.T).T + np.longlong(
            np.sum((np.dot(x, self.v) ** 2 - np.dot(x ** 2, self.v ** 2)),
                   axis=1).reshape(len(x), 1)) / 2.0

        return z, sigmoid(z)

    def evaluate(self):
        """
        在验证集上计算R2
        :return:
        """
        _, y_hat = self.predict(self.valid_x)
        mse = np.sum((y_hat - self.valid_y) ** 2) / len(self.valid_y)
        r2 = 1.0 - mse / self.var_y
        print("r2={}".format(r2))
        return r2

    def update_mini_batch(self, x, y, eta):
        """
        平方误差作为损失函数，梯度下降法更新参数
        :param x:
        :param y:
        :param eta: 学习率
        :return:
        """
        batch = len(x)
        step = eta / batch
        z, y_hat = self.predict(x)
        y_diff = self.cost.delta(z, y_hat, y)
        self.w0 -= step * np.sum(y_diff)
        self.w -= step * np.dot(y_diff.T, x)
        delta_v = np.zeros(self.v.shape)
        for i in range(batch):
            xi = x[i:i + 1, :]  # mini_batch中的第i个样本。为保持shape不变，注意这里不能用x[i]
            delta_v += (np.outer(xi, np.dot(xi, self.v)) - xi.T ** 2 * self.v) * (y_diff[i])
        self.v -= step * delta_v

    def train(self, mini_batch=100):
        """
        采用批量梯度下降法训练模型
        :param mini_batch:
        :return:
        """
        for itr in range(self.maxecho):
            print("iteration={}".format(itr))
            self.shuffle_data()
            n = len(self.train_x)
            for b in range(0, n, mini_batch):
                x = self.train_x[b:b + mini_batch]
                y = self.train_y[b:b + mini_batch]
                learn_rate = np.exp(-itr) * self.eta  # 学习率指数递减
                self.update_mini_batch(x, y, learn_rate)

            if self.evaluate() > self.r2:
                break


def fake_data(sample, dim, k):
    """
    构造假数据
    :param sample:
    :param dim:
    :param k:
    :return:
    """
    w0 = np.random.randn()
    w = np.random.randn(1, dim)
    v = np.random.randn(dim, k)
    x = np.random.randn(sample, dim)
    z = w0 + np.dot(w, x.T).T + np.longlong(
        np.sum((np.dot(x, v) ** 2 - np.dot(x ** 2, v ** 2)),
               axis=1).reshape(len(x), 1)) / 2.0
    y = sigmoid(z)
    data = np.concatenate((x, y), axis=1)
    return z, data


if __name__ == "__main__":
    dim = 9  # 特征的维度
    k = dim / 3
    sample = 100
    z, data = fake_data(sample, dim, k)

    train_size = int(0.7 * sample)
    valid_size = int(0.2 * sample)
    train = data[:train_size]  # 训练集
    valid = data[train_size:train_size + valid_size]  # 验证集
    test = data[train_size + valid_size:]  # 测试集
    test_z = z[train_size + valid_size:]

    eta = 0.01  # 初始学习率
    maxecho = 200
    r2 = 0.9  # 拟合系数r2的最小值
    fm = FM(train, valid, k, eta, maxecho, r2)
    fm.train(mini_batch=50)

    test_x = test[:, :-1]
    test_y = test[:, -1:]
    print('z=', test_z)
    print("y=", test_y)
    z_hat, y_hat = fm.predict(test_x)
    print("z_hat=", z_hat)
    print("y_hat=", y_hat)