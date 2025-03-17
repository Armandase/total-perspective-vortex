from CustomEstimator import CustomEstimator
from CustomTransformer import CustomTransformer
from sklearn.pipeline import Pipeline

def create_pipeline(reducter_name='csp', estimator_name='lda'):
    # check if reducter name is a list
    transformer = []
    reducter_name = ['WAVELET', reducter_name]
    # reducter_name = [ reducter_name]
    if isinstance(reducter_name, list):
        for name in reducter_name:
            transformer.append((name, CustomTransformer(name=name)))
    else:
        transformer.append((reducter_name, CustomTransformer(name=reducter_name)))
    estimator = [(estimator_name, CustomEstimator(name=estimator_name))]
    
    pipeline_list = transformer + estimator
    clf = Pipeline(pipeline_list)
    return clf