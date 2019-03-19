import numpy as np

class StandardScaler:
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
               "must fit before transform!"
        assert X.shape[1] == len(self.mean_), \
               "the feature number of X must be equal to mean_ and scala_"

        scaledX = np.empty(shape=X.shape, dtype=float)    
        for col in range(X.shape[1]):
            scaledX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]

        return scaledX

class MinMaxScaler:
    """暂时未实现有feature_range参数的版本，默认数据范围[0 , 1]"""
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        self.min_ = np.array([np.min(X[:, i]) for i in range(X.shape[1])])
        self.max_ = np.array([np.max(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.min_ is not None and self.max_ is not None, \
               "must fit before transform!"
        assert X.shape[1] == len(self.min_), \
               "the feature number of X must be equal to mean_ and scala_"
        scaledX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            scaledX[:,col] = (X[:col] - self.min_[col]) / (self.max_[col] - self.min_[col])

        return scaledX    