import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import accuracy_score

# estimator is used to predict the data
class CustomEstimator(BaseEstimator):
    def __init__(self, *, name=''):
        super().__init__()
        self.name = name
        self.is_fitted_ = False
        self.classifier = self.select_estimator(name)
    def select_estimator(self, name: str):
        name = name.upper()
        classifier = None
        if name == "LDA":
            # LDA creates a hyperplane that separates the data
            # it maximizethe distance between the means for the n categories
            # and minimizes the variance within each category
            classifier = LinearDiscriminantAnalysis(n_components=None,
                                                priors=None,
                                                shrinkage=None,
                                                solver="svd",
                                                store_covariance=False,
                                                tol=0.0001)
        elif name == "LDA_SHRINKAGE":
            classifier = LinearDiscriminantAnalysis(n_components=None,
                                                priors=None,
                                                shrinkage="auto",
                                                solver="lsqr",
                                                store_covariance=False,
                                                tol=0.0001)
        elif name == "SVM":
            classifier = svm.SVC()
        else:
            raise Exception(f'{name} is not handle')
        return classifier
    
    def fit(self, X, y):
        # Reshape X to be 2-dimensional
        # n_samples, n_channels, n_times = X.shape
        # X_reshaped = X.reshape(n_samples, n_channels * n_times)
        self.classifier.fit(X=X, y=y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        if not self.is_fitted_:
            raise Exception("This %s instance is not fitted yet" % self.__class__.__name__)
        # Reshape X to be 2-dimensional
        # n_samples, n_channels, n_times = X.shape
        # X_reshaped = X.reshape(n_samples, n_channels * n_times)
        # X_reshaped = X.reshape(n_samples, n_channels * n_times)
        y_pred = self.classifier.predict(X)
        return y_pred
        
    def score(self, X, y_true):
        y_pred = self.predict(X=X)
        return accuracy_score(y_true=y_true, y_pred=y_pred)    
    
    def get_name(self):
        return self.name