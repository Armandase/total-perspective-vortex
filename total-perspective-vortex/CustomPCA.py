import numpy as np
import scipy
from sklearn.base import TransformerMixin


# CSP involves transforming the EEG data from the time domain to the spatial domain
#   using a spatial filter. It applies a spatial filter to the multi-channel EEG data
#   to enhance the signal variance of one class while reducing it for the other within the same domain.
class CustomPCA(TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.weights = None
        self.name = 'CustomPCA'
        self.is_fit_ = False
        
    def normalized_spatial_covariance(self, X):
        cov = np.dot(X, X.T)
        return cov / np.trace(cov)
    
    def population_covariance(self, X):
        n_trials = X.shape[0]
        factor = 1 / n_trials
        pop_cov = np.zeros((X.shape[1], X.shape[1]))
        for trial in X:
            cov = self.normalized_spatial_covariance(trial)
            pop_cov += factor * cov
        return pop_cov
    
    def extract_class(self, X, y):
        list_class = []

        for i in range(len(np.unique(y))):
            list_class.append(X[y == i])
        return list_class
    
    def decentre_data(self, X, X_avg):
        return X - X_avg
    
    def fit(self, X, y=None):
        X_cpy = X.copy()

        decentered_data = [self.decentre_data(trial, np.mean(trial, axis=0)) for trial in X_cpy]
        decentered_data = np.asarray(decentered_data)
        
        norm_spatial_cov = self.population_covariance(decentered_data)
        
        eigvals, eigvecs = scipy.linalg.eigh(norm_spatial_cov)
        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs, axis=1)

        self.weights = eigvecs[:, :self.n_components]
        
        X = np.asarray([np.dot(self.weights.T, x) for x in X])
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        self.is_fit_ = True
        return self

    def transform(self, X, log=False):
        if self.is_fit_ is False:
            raise ValueError("The model is not fitted yet")
        X_transform = X.copy()

        # X_transform = [self.decentre_data(trial, np.mean(trial, axis=0)) for trial in X_transform]
        # X_transform = np.asarray(X_transform)
        # apply spatial filter to the data
        X_transform = np.asarray([np.dot(self.weights.T, x) for x in X_transform])
        X_transform = (X_transform ** 2).mean(axis=2)
        if log is True:
            X_transform = np.log(X_transform)
        else:
            X_transform -= self.mean
            X_transform /= self.std
        return X_transform
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_name(self):
        return self.name