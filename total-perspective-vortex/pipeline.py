from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import TransformerMixin, BaseEstimator

def create_pipeline(reducter_name='csp', classifier_name='lda'):
    reducter_name = reducter_name.upper()
    classifier_name = classifier_name.upper()
    classifier = None
    reducter = None

    if reducter_name == 'CSP':
        reducter = CSP(n_components=4,
                       reg=None,
                       log=True,
                       norm_trace=False,
                       cov_est="concat",
                       cov_method_params=None,
                       transform_into="average_power", rank=None)
    if classifier_name == "LDA":
        # LDA creates a hyperplane that separates the data
        # it maximizethe distance between the means for the n categories
        # and minimizes the variance within each category
        classifier = LinearDiscriminantAnalysis(n_components=None,
                                            priors=None,
                                            shrinkage=None,
                                            solver="svd",
                                            store_covariance=False,
                                            tol=0.0001)
    elif classifier_name == "LDA_SHRINKAGE":
        classifier = LinearDiscriminantAnalysis(n_components=None,
                                            priors=None,
                                            shrinkage="auto",
                                            solver="lsqr",
                                            store_covariance=False,
                                            tol=0.0001)

    clf = Pipeline([(reducter_name, reducter),
                    (classifier_name, classifier)])
    return clf