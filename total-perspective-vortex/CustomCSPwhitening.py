import numpy as np
import scipy
from sklearn.base import TransformerMixin


# CSP involves transforming the EEG data from the time domain to the spatial domain
#   using a spatial filter. It applies a spatial filter to the multi-channel EEG data
#   to enhance the signal variance of one class while reducing it for the other within the same domain.
class CustomCSPwhitening(TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters = None
        self.name = 'customCSPWhitening'
        self.is_fit_ = False
        
    def normalized_spatial_covariance(self, X):
        """
        compute the covariance for a single trial
            and normalize it to eliminate the magnitude variation in EEG between individuals
        
         input: X (n_channels, n_samples)
         
         return: cov (n_channels, n_channels)
        """
        cov = np.dot(X, X.T)
        return cov / np.trace(cov)
    
    def population_covariance(self, X):
        """
            compute the average spatial covariance
        """ 
        
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
        """
        Decenter the data with respect to their class average (recentre to the origin)
        """
        return X - X_avg
    
    def fit(self, X_in, y=None):
        X = X_in.copy()
        list_class = self.extract_class(X, y)
        avg_class = []
        for x_class in list_class:
            avg_class.append(np.mean(x_class, axis=0))
        for i in range(X.shape[0]):
            X[i] = self.decentre_data(X[i], avg_class[y[i]]) # En
        
        decentred_list_class = self.extract_class(X, y)
        norm_spatial_cov = []
        for decentred_class in decentred_list_class:
            norm_spatial_cov.append(self.population_covariance(decentred_class)) # (Cn)
        
        #compostite qui signifie que c'est la combinaison de plusieurs elements
        composite_spatial_cov = np.sum(norm_spatial_cov, axis=0) # (Cc)
        # composite_spatial_cov = np.concatenate(norm_spatial_cov) # (Cc)
        # decomposition en valeurs propres
        # les valeurs propres sont egales aux racines du polynome caracteristique de la matrice
        eigvals, eigvecs = scipy.linalg.eigh(composite_spatial_cov) # (V)


        diag_inv_sqrt = np.diag(np.sqrt(1/(eigvals + 1e-6))) # (D)
        
        # whitening matrix (P)
        whitening = np.dot(diag_inv_sqrt, eigvecs.T)
        
        # whitening performed on the spatial covariance matrices
        spatial_cov_whitened = [] # Sn = PCnP′
        for cov in norm_spatial_cov:
            spatial_cov_whitened.append(np.dot(np.dot(cov, diag_inv_sqrt.T), whitening)) 

        composite_spatial_cov_whitened = np.sum(spatial_cov_whitened, axis=0) # (BΛnB′)
        eigvals, eigvecs = scipy.linalg.eigh(composite_spatial_cov_whitened) # (V)
        
        # To standardize features
        self.filters = eigvecs.T
        
        X_in = np.asarray([np.dot(self.filters, x) for x in X_in])
        X_in = (X_in ** 2).mean(axis=2)

        # To standardize features
        self.mean = X_in.mean(axis=0)
        self.std = X_in.std(axis=0)
        
        self.is_fit_ = True
        return self

    def transform(self, X, log=False):
        X_transform = X.copy()
        # apply spatial filter to the data

        if self.is_fit_ is False:
            # raise ValueError("The model is not fitted yet")
            print("The model is not fitted yet")
            return X_transform
        X_transform = np.asarray([np.dot(self.filters, x) for x in X_transform])
        X_transform = (X_transform ** 2).mean(axis=2)
        if log is True:
            X_transform = np.log(X_transform)
        else:
            X_transform -= self.mean
            X_transform /= self.std
        return X_transform
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        # filters should be (64, 64)
        # output should be (12, n_components)
        return self.transform(X)
    
    def get_name(self):
        return self.name