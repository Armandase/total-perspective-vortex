import numpy as np
from mne.decoding import CSP
from sklearn.base import TransformerMixin

# transformer is used to transform the data but did apply any learning
class CustomTransformer(TransformerMixin):
    def __init__(self, *, name=''):
        super().__init__()
        self.name = name
        self.transformer = self.select_transformer(name)

    def select_transformer(self, name: str):
        name = name.upper()
        transformer = None
        if name == "CSP":
            transformer = CSP(n_components=4,
                reg=None,
                log=True,
                norm_trace=False,
                cov_est="concat",
                cov_method_params=None,
                transform_into="average_power", rank=None)
        else:
            raise Exception(f'{name} is not handle')
        return transformer

    def crop_data(self, X):
        epochs_train = X.copy().crop(tmin=1.0, tmax=2.0)
        epochs_data_train = epochs_train.get_data(copy=False)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)
    
    def fit_transform(self, X, y):
        return self.transformer.fit_transform(X, y)
    
    def get_name(self):
        return self.name