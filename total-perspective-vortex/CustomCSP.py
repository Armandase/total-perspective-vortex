import numpy as np
import scipy
from sklearn.base import TransformerMixin


# CSP involves transforming the EEG data from the time domain to the spatial domain
#   using a spatial filter. It applies a spatial filter to the multi-channel EEG data
#   to enhance the signal variance of one class while reducing it for the other within the same domain.
class CustomCSP(TransformerMixin):
    def __init__(self, n_components=4, cov_est='avg'):
        self.n_components = n_components
        self.filters = None
        self.name = 'CustomCSP'
        self.cov_est = cov_est
        self.is_fit_ = False
    
    def decentre_data(self, X, X_avg):
        """
        Decenter the data with respect to their class average
        """
        return X - X_avg
        
    def normalized_spatial_covariance(self, X):
        """
        compute the covariance for a single trial
            and normalize it to eliminate the magnitude variation in EEG between individuals
        
         input: X (n_channels, n_samples)
         
         return: cov (n_channels, n_channels)
        """
        cov = np.dot(X, X.T)
        return cov / np.trace(cov)
    
    def covariance_np(self, X, ddof=0):
        """Compute the covariance for a single trial based on the numpy implementation

        Args:
            X (np.ndarray): input data (n_channels, n_samples)
            ddof (int, optional): 1 return the unbiased estimate. 0 return the average. Defaults to 0.

        Returns:
            np.ndarray: covariance matrix (n_channels, n_channels)
        """
        X -= np.mean(X, axis=1)[:, None]
        fact = X.shape[1] - ddof
        return np.dot(X, X.T.conj()) * np.true_divide(1, fact)
    
    
    def extract_class(self, X, y):
        list_class = []

        for i in range(len(np.unique(y))):
            list_class.append(X[y == i])
        return list_class

    def avg_covariance(self, X, type='np'):
        """Compute the avg covariance for a batch

        Args:
            X (np.ndarray): input data (n_trials, n_channels, n_samples)

        Returns:
            np.ndarray: covariance matrix (n_channels, n_channels)
        """
        n_trials = X.shape[0]
        factor = 1 / n_trials
        pop_cov = np.zeros((X.shape[1], X.shape[1]))
        for trial in X:
            if type == 'np':
                cov = self.covariance_np(trial)
            else:
                cov = self.normalized_spatial_covariance(trial)
            pop_cov += factor * cov
        return pop_cov
    
    def concat_covariance(self, X, type='np'):
        """Compute the covariance for a batch

        Args:
            X (np.ndarray): input data (n_trials, n_channels, n_samples)

        Returns:
            np.ndarray: covariance matrix (n_channels, n_channels)
        """
        _, n_channels, _ = X.shape
        x_class = X.transpose(1, 0, 2).reshape(n_channels, -1)
        if type == 'np':
            cov = self.covariance_np(x_class)
        else:
            cov = self.normalized_spatial_covariance(x_class)
        return cov / np.trace(cov)
        
    def fit(self, X, y=None):
        X_cpy = X.copy()
        list_class = self.extract_class(X_cpy, y)
        avg_class = []
        for x_class in list_class:
            avg_class.append(np.mean(x_class, axis=0))
        for i in range(X_cpy.shape[0]):
            X_cpy[i] = self.decentre_data(X_cpy[i], avg_class[y[i]]) # En
        
        decentred_list_class = self.extract_class(X_cpy, y)
        norm_spatial_cov = []
        for decentred_class in decentred_list_class:
            if self.cov_est == 'avg':
                norm_spatial_cov.append(self.avg_covariance(decentred_class))
            else:
                norm_spatial_cov.append(self.concat_covariance(decentred_class)) # (Cn)
        
        # composite qui signifie que c'est la combinaison de plusieurs elements
        composite_spatial_cov = np.sum(norm_spatial_cov, axis=0) # (Cc)
        # decomposition en valeurs propres
        # les valeurs propres sont egales aux racines du polynome caracteristique de la matrice
        # sorted in ascending order        
        eigvals, eigvecs = scipy.linalg.eigh(composite_spatial_cov) # (V)
        
        self.filters = eigvecs

        picked_filters = self.filters[:, :self.n_components].T
        X = np.asarray([np.dot(picked_filters, x) for x in X])
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        self.is_fit_ = True
        return self

    def transform(self, X, log=False):
        X_transform = X.copy()

        if self.is_fit_ is False:
            raise ValueError("The model is not fitted yet")
        
        # apply spatial filter to the data
        picked_filters = self.filters[:, :self.n_components].T
        X_transform = np.asarray([np.dot(picked_filters, x) for x in X_transform])
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