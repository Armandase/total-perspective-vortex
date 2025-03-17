import numpy as np
import scipy
from sklearn.base import TransformerMixin


# CSP involves transforming the EEG data from the time domain to the spatial domain
#   using a spatial filter. It applies a spatial filter to the multi-channel EEG data
#   to enhance the signal variance of one class while reducing it for the other within the same domain.
class CustomCSP(TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters = None
        self.name = 'CustomCSP'
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
    
    def fit(self, X, y=None):
        print('Fitting CSP')
        X = X.copy()
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
        # decomposition en valeurs propres
        # les valeurs propres sont egales aux racines du polynome caracteristique de la matrice
        # eigvals, eigvecs = np.linalg.eig(composite_spatial_cov) # (V)
        eigvals, eigvecs = scipy.linalg.eigh(composite_spatial_cov) # (V)
        # diagonal matrix of eigenvalues sorted in descending order
        # sorted_eigvals = np.sort(eigvals)[::-1]
        diag = np.diag(eigvals) # (D)
        print('eigvecs shape: ', eigvecs.shape)
        
        # whitening matrix (P)
        whitening = np.dot(np.sqrt(np.linalg.inv(diag)), eigvecs.T)
        
        # whitening performed on the spatial covariance matrices
        spatial_cov_whitened = [] # PCnP′
        for cov in norm_spatial_cov:
            spatial_cov_whitened.append(np.dot(np.dot(whitening, cov), whitening.T))

        whitening_eigvecs = [] # BΛnB′
        for cov in spatial_cov_whitened:
            eigvals, eigvecs = scipy.linalg.eigh(cov)
            # sorted_eigvals = np.sort(eigvals)[::-1]
            whitening_eigvecs.append(eigvals)
        
        # spatial filter (W)
        spatial_filter = []
        for eigvecs in whitening_eigvecs:
            # spatial_filter.append(np.dot(eigvecs.T, whitening))
            spatial_filter.append(eigvecs.T)

        self.filters = np.array(spatial_filter)
        # self.filters = np.array(spatial_filter[:self.n_components])
        
        self.is_fit_ = True
        return self

    def transform(self, X):
        print('Transform CSP')
        X_transform = X.copy()
        # apply spatial filter to the data

        if self.is_fit_ is False:
            # raise ValueError("The model is not fitted yet")
            print("The model is not fitted yet")
            return X_transform
        print(self.filters.shape)
        X_transform = np.asarray([np.dot(self.filters, x) for x in X_transform])
        X_transform = (X_transform ** 2).mean(axis=2)
        print(X_transform.shape)
        return X_transform
    
    def fit_transform(self, X, y):
        print('Fitting and transforming CSP')
        self.fit(X, y)
        # filters should be (64, 64)
        # output should be (12, n_components)
        return self.transform(X)
    
    def get_name(self):
        return self.name