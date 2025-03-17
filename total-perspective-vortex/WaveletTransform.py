from sklearn.base import TransformerMixin
import pywt
import numpy as np

# def maddest(d, axis=None):
#     return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# def denoise(x, wavelet='haar', level=1):
#     ret = {key:[] for key in x.columns}
    
#     for pos in x.columns:
#         coeff = pywt.wavedec(x[pos], wavelet, mode="per")
#         sigma = (1/0.6745) * maddest(coeff[-level])

#         uthresh = sigma * np.sqrt(2*np.log(len(x)))
#         coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

#         ret[pos]=pywt.waverec(coeff, wavelet, mode='per')
    
#     return pd.DataFrame(ret)



class WaveletTransform(TransformerMixin):
    def __init__(self, *, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.name = 'WaveletTransform'
        self.is_fit_ = False

    def maddest(self, d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def denoise(self, X, wavelet='db1', level=1):
        """Denoise the data using wavelet transform

        Args:
            X (np.array): raw data to be denoised shap should be (n_samples, n_channels, n_times)
            wavelet (str, optional): Wavelet to use. Defaults to 'db1'.
            level (int, optional): decomposition level. Defaults to 1.
        """
        ret = np.zeros((X.shape[0], X.shape[1], X.shape[2] + 1))
        factor = 0.6745
        threshold = np.sqrt(2*np.log(len(X)))
        for i, trial in enumerate(X):
            coeff = pywt.wavedec(data=trial, wavelet=wavelet, level=level)
            sigma = (1/factor) * self.maddest(coeff[-level])
            uthresh = sigma * threshold
            coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
            
            ret[i] = pywt.waverec(coeff, wavelet, mode='per')
        return ret        

    def fit(self, X, y=None):
        self.is_fit_ = True
        return self

    def transform(self, X):
        X_transform = X.copy()

        if self.is_fit_ is False:
            raise ValueError("The model is not fitted yet")
        X_transform = self.denoise(X_transform)
        return X_transform
    
    def fit_transform(self, X, y):
        print('Fitting and transforming ', self.name)
        self.fit(X, y)
        # filters should be (64, 64)
        # output should be (12, n_components)
        return self.transform(X)
    
    def get_name(self):
        return self.name