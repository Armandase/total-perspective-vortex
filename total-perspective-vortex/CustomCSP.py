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
        
    def spatial_covariance(self, X):
        """
        compute the covariance for a single trial
            and normalize it to eliminate the magnitude variation in EEG between individuals
        
         input: X (n_channels, n_samples)
         
         return: cov (n_channels, n_channels)
        """
        cov = X.T @ X
        return cov / np.trace(cov)
    
    def population_covariance(self, X):
        """
            compute the average spatial covariance
        """ 
        
        n_trials = X.shape[0]
        factor = 1 / n_trials
        pop_cov = np.zeros((X.shape[1], X.shape[1]))
        for trial in X:
            cov = self.spatial_covariance(trial)
            pop_cov += factor * cov
        return pop_cov
    
    def extract_class(self, X, y):
        list_class = []
        print()
        for i in np.unique(y):
            list_class.append(X[y == i])
        print(list_class)
        exit(0)
        return list_class
    
    def fit(self, X, y=None):
        
        for batch in range(X.shape[0]):
            x = X[batch]
            
            # sigma = x @ x.T
            # trace = np.trace(x)
            R1 = X[0] @ X[0].T
            R2 = X[1] @ X[1].T
            R1 = R1 / x.shape[1]
            R2 = R2 / x.shape[1]
            
            # eigvals, eigvecs = np.linalg.eig(R1, R2)
            eigvals, eigvecs = np.linalg.eig(R1)
            eigvals2 = np.diagonal(np.dot(np.dot(eigvecs.T, R2), eigvecs))
            print(eigvals)
            print(eigvals2)
            # matrix of eigenvectors P = [p1 ... pn]
            # the diagonal matrix D {\displaystyle \mathbf {D} } of eigenvalues { λ 1 , ⋯ , λ n } {\displaystyle \{\lambda _{1},\cdots ,\lambda _{n}\}} sorted by decreasing order such that
            D = np.diag(eigvals)
            D2 = np.diag(eigvals2)
            print(D)
            print(D2)
            
            exit(0)
            
            
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


        return X
    
    def fit_transform(self, X, y):
        X = self.transform(X.copy())
        return self.fit(X, y)
    
    def get_name(self):
        return self.name