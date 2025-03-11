import numpy as np
from sklearn.base import BaseEstimator

# estimator is used to predict the data
class CustomEstimator(BaseEstimator):
    def __init__(self, *, param=1):
        super().__init__()
        self.param = param
    def fit(self, X, y):
        self.is_fitted_ = True
        return self
    def predict(self, X):
        if not self.is_fitted_:
            raise Exception("This %s instance is not fitted yet" % self.__class__.__name__)
        return np.zeros(X.shape[0])