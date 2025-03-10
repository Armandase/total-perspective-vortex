# exemple
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# X, y = make_classification(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     random_state=0)
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# # The pipeline can be used as any other estimator
# # and avoids leaking the test set into the train set
# pipe.fit(X_train, y_train).score(X_test, y_test)
# # An estimator's parameter can be set using '__' syntax
# pipe.set_params(svc__C=10).fit(X_train, y_train).score(X_test, y_test)

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
                       transform_into="average_power")
    if classifier_name == "LDA":
        classif = LinearDiscriminantAnalysis(n_components=None,
                                            priors=None,
                                            shrinkage=None,
                                            solver="svd",
                                            store_covariance=False,
                                            tol=0.0001)

    clf = Pipeline([(reducter_name, reducter),
                    (classifier_name, classifier)])
    return clf