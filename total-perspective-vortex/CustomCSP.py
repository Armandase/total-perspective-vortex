import numpy as np
from sklearn.base import TransformerMixin


# CSP involves transforming the EEG data from the time domain to the spatial domain
#   using a spatial filter. It applies a spatial filter to the multi-channel EEG data
#   to enhance the signal variance of one class while reducing it for the other within the same domain.
class CustomCSP(TransformerMixin):
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False, cov_est="concat", cov_method_params=None, transform_into="average_power", rank=None):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.cov_est = cov_est
        self.cov_method_params = cov_method_params
        self.transform_into = transform_into
        self.rank = rank
        self.name = 'CustomCSP'
    
    def time_avg_covariance(self, X):
        # n_trials, n_channels, n_samples = X.shape
        # cov = np.zeros((n_channels, n_channels))
        # for i in range(n_trials):
        #     cov += np.dot(X[i], X[i].T)
        # return cov / n_trials
        
        cov = X @ X.T
        
        return cov / X.shape[1]
        
    def normalized_spatial_covariance(self, X):
        """
        compute the covariance for a single trial
            and normalize it to eliminate the magnitude variation in EEG between individuals
        
         input: X (n_channels, n_samples)
         
         return: cov (n_channels, n_channels)
        """
        # cov = np.dot(X.T, X)
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
        Decenter the data with respect to their class average
        """
        return X - X_avg
    
    def fit(self, X, y=None):
        X = X.copy()
        list_class = self.extract_class(X, y)
        avg_class = []
        for x_class in list_class:
            avg_class.append(np.mean(x_class, axis=0))
        for i in range(X.shape[0]):
            X[i] = self.decentre_data(X[i], avg_class[y[i]])
        
        decentred_list_class = self.extract_class(X, y)
        norm_spatial_cov = []
        for decentred_class in decentred_list_class:
            norm_spatial_cov.append(self.population_covariance(decentred_class))
        
        #compostite qui signifie que c'est la combinaison de plusieurs elements
        composite_spatial_cov = np.sum(norm_spatial_cov, axis=0)
        # decomposition en valeurs propres
        # les valeurs propres sont egales aux racines du polynome caracteristique de la matrice
        eigvals, eigvecs = np.linalg.eig(composite_spatial_cov)
        # diagonal matrix of eigenvalues sorted in descending order
        sorted_eigvals = np.sort(eigvals)[::-1]
        diag = np.diag(sorted_eigvals)
        
        print('Diagonal:', diag.shape)
        print('Eigvals:', eigvals.shape)
        print('eigvecs:', eigvecs.shape)
        
        # whitening matrix
        whitening = np.dot(np.sqrt(np.linalg.inv(diag)), eigvecs.T)
        print('Whitening:', whitening.shape)
        
        # whitening performed on the spatial covariance matrices
        spatial_cov_whitened = []
        for cov in norm_spatial_cov:
            spatial_cov_whitened.append(np.dot(np.dot(whitening, cov), whitening.T))

        whitening_eigvecs = []
        # whitening_diags = []
        for cov in spatial_cov_whitened:
            eigvals, eigvecs = np.linalg.eig(cov)
            sorted_eigvals = np.sort(eigvals)[::-1]
            whitening_eigvecs.append(eigvecs)
            # whitening_diags.append(np.diag(sorted_eigvals))
        
        # spatial filter
        spatial_filter = []
        for eigvecs in whitening_eigvecs:
            spatial_filter.append(np.dot(eigvecs.T, whitening))

        # apply spatial filter to the data
        # X_filtered = []
        # for i in range(X.shape[0]):
        #     X_filtered.append(np.dot(spatial_filter[y[i]], X[i]))
        #     print('X_filtered:', X_filtered[-1].shape)
        # exit(1)
        return self

    def transform(self, X):
        # Xcsp(t) = Wtranspos * x(t)
        
        # Xcsp = AS 
        # ou    A est une matrice de mélange qui contient
        #           les coefficients indiquant comment chaque source
        #           contribue aux des dimensions de l'observation
        #       S correspond aux signaux sources
        
        # On condisere que chaque colonne de X et S sont des samples temporelles:
        # x(t) = As(t)
        
        # La matrice A joue un role essentiel dans la maniere
        #   dont les statitques des signaux obesrvés x(t) se retrouvent
        #   liées a celle des sources originales s(t)
        # L’effet de la matrice de mélange A peut être vu comme  transformation
        #   qui prend les statistiques simples (diagonales) des sources et 
        #   les « étend » pour expliquer les statistiques croisées plus riches observées dans x(t).
        # On trouve donc l'equation suivante:
        # x(t)x(t)transpos = As(t) * (As(t))transpos
        #                  = As(t) * s(t)transpos * Atranspos
        #                  = ARsAtranspos

        if self.is_fitted_ is False:
            raise Exception("Model not fitted")
        X = X.copy()
        
        
        return X
    
    def fit_transform(self, X, y):
        X = self.transform(X.copy())
        return self.fit(X, y)
    
    def get_name(self):
        return self.name