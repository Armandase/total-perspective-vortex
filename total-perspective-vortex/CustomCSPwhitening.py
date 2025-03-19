import numpy as np
import scipy
from CustomCSP import CustomCSP


# CSP involves transforming the EEG data from the time domain to the spatial domain
#   using a spatial filter. It applies a spatial filter to the multi-channel EEG data
#   to enhance the signal variance of one class while reducing it for the other within the same domain.
class CustomCSPwhitening(CustomCSP):
    def __init__(self, n_components=4, cov_est='concat'):
        super().__init__(n_components, cov_est)
        self.name = 'customCSPWhitening'

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
            if self.cov_est == 'avg':
                norm_spatial_cov.append(self.avg_covariance(decentred_class).T)
            else:
                norm_spatial_cov.append(self.concat_covariance(decentred_class).T) # (Cn)
        
        #compostite qui signifie que c'est la combinaison de plusieurs elements
        composite_spatial_cov = np.sum(norm_spatial_cov, axis=0) # (Cc)
        # composite_spatial_cov = np.concatenate(norm_spatial_cov) # (Cc)
        # decomposition en valeurs propres
        # les valeurs propres sont egales aux racines du polynome caracteristique de la matrice
        eigvals, eigvecs = scipy.linalg.eigh(composite_spatial_cov) # (V)

        diag_inv_sqrt = np.diag(np.sqrt(1/(eigvals + 1e-6))) # (D)
        
        # whitening matrix (P)
        whitening = np.dot(diag_inv_sqrt, eigvecs.T)
        self.whitening_filter = whitening
        
        # whitening performed on the spatial covariance matrices
        spatial_cov_whitened = [] # Sn = PCnP′
        for cov in norm_spatial_cov:
            spatial_cov_whitened.append(np.dot(np.dot(cov, whitening.T), whitening)) 

        composite_spatial_cov_whitened = np.sum(spatial_cov_whitened, axis=0) # (BΛnB′)
        eigvals, eigvecs = scipy.linalg.eigh(spatial_cov_whitened[0], composite_spatial_cov_whitened) # (V)
        # To standardize features
        # self.filters = np.dot(eigvecs.T, self.whitening_filter)
        self.filters = eigvecs
        
        picked_whitening = self.whitening_filter.T
        picked_filters = self.filters[:, :self.n_components].T
        X_in = np.asarray([np.dot(picked_whitening, x) for x in X_in])
        X_in = np.asarray([np.dot(picked_filters, x) for x in X_in])        
        X_in = (X_in ** 2).mean(axis=2)

        # To standardize features
        self.mean = X_in.mean(axis=0)
        self.std = X_in.std(axis=0)
        
        self.is_fit_ = True
        return self
    
    
    def transform(self, X, log=False):
        X_transform = X.copy()

        if self.is_fit_ is False:
            raise ValueError("The model is not fitted yet")
        
        # apply spatial filter to the data
        picked_whitening = self.whitening_filter.T
        picked_filters = self.filters[:, :self.n_components].T
        X_transform = np.asarray([np.dot(picked_whitening, x) for x in X_transform])        
        X_transform = np.asarray([np.dot(picked_filters, x) for x in X_transform])        
        X_transform = (X_transform ** 2).mean(axis=2)
        if log is True:
            X_transform = np.log(X_transform)
        else:
            X_transform -= self.mean
            X_transform /= self.std
        return X_transform