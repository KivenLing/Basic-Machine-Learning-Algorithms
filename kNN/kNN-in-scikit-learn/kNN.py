import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        self.k = k
        self._X_train = None
        self._y_train = None

    """
    为了符合scikit learn的算法流程
    需要有两个一般过程方法: fit 和 predict
    """    
    
    def fit(self, X_train, y_train):
        # X_train 为包含特征的训练集
        # y_train 为上述训练集所属的类别
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        # 必须先进行fit
        # X_predict 多组待预测的特征向量
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)


    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]
        
    def __repr__(self):
        return "KNN(k=%d)" % self.k