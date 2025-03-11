import numpy as np
from sklearn.base import TransformerMixin

# transformer is used to transform the data but did apply any learning
class CustomTransformer(TransformerMixin):
    def __init__(self, *, param=1):
        super().__init__()
        self.param = param

    def crop_data(self, X):
        epochs_train = X.copy().crop(tmin=1.0, tmax=2.0)
        epochs_data_train = epochs_train.get_data(copy=False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.crop_data(X)