from CustomEstimator import CustomEstimator
from CustomTransformer import CustomTransformer
from sklearn.pipeline import Pipeline

def create_pipeline(reducter_name='csp', estimator_name='lda'):
    transformer = CustomTransformer(name=reducter_name)
    estimator = CustomEstimator(name=estimator_name)
    
    clf = Pipeline([(reducter_name, transformer),
                    (estimator_name, estimator)])
    return clf