from CustomEstimator import CustomEstimator
from CustomTransformer import CustomTransformer
from sklearn.pipeline import Pipeline

def create_pipeline(pipeline_names):
    # check if reducter name is a list
    transformer_names = pipeline_names[:-1]
    estimator_name = pipeline_names[-1]
    transformer = []
    for name in transformer_names:
        transformer.append((name, CustomTransformer(name=name)))
    estimator = [(estimator_name, CustomEstimator(name=estimator_name))]
    
    pipeline_list = transformer + estimator
    clf = Pipeline(pipeline_list)
    return clf