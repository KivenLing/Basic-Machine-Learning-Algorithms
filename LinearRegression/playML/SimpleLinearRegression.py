import numpy as np

class SimpleLinearRegression:
    """这里先实现一维线性回归，即利用最小二乘法得到的结果"""
    def __init__(self):
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_test):
        """给定待预测数据集x_test，返回表示x_test的结果向量"""
        
        return np.array([self._predict(x) for x in x_test])

    def _predict(self, x_single):
        """给定单个待预测数据x_single，返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()"