import numpy as np
from mne.decoding import CSP, UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
from sklearn.base import TransformerMixin
from CustomCSP import CustomCSP
from CustomCSPwhitening import CustomCSPwhitening
from CustomPCA import CustomPCA
from WaveletTransform import WaveletTransform


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
        elif name == 'CUSTOM_CSP':
            transformer = CustomCSP(n_components=64)
        elif name == 'CUSTOM_CSP_WHITENING':
            transformer = CustomCSPwhitening(n_components=64)
        elif name == 'WAVELET':
            transformer = WaveletTransform(n_channels=64)
        elif name == "CUSTOM_PCA":
            transformer = CustomPCA(n_components=10)
        else:
            raise Exception(f'{name} is not handle')
        return transformer

    def crop_data(self, X):
        epochs_raw_data = X.copy().crop(tmin=1.0, tmax=2.0)
        epochs_data = epochs_raw_data.get_data(copy=False)
        return epochs_data

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)
    
    def fit_transform(self, X, y):
        return self.transformer.fit_transform(X, y)
    
    def get_name(self):
        return self.name