print("Hello, World!")
from collections import defaultdict

class ModelMetrics(object):

    # Random permutation cross-validator with 80-20 train test split
    cv = ShuffleSplit(n_splits = 2, test_size = 0.2) # change this to 10
    # dictionary to store scores
    model_scores = defaultdict(lambda:None)
    # default scoring metrics
    default_metric = ['accuracy','precision', 'recall']

    def __init__(self, model_name, model_obj, features, response):
        self.model = model_name
        self.model_obj = model_obj
        ModelMetrics.model_scores[model_name] = []
        self.cv = ModelMetrics.cv
        self.features = features
        self.response = response
        self.model_scores = ModelMetrics.model_scores

    def model_scoring(self, scoring_metric=default_metric):
        for metric in scoring_metric:
            n_fold_score = cross_val_score(self.model_obj,self.features,
                                                           self.response,
                                                           cv=self.cv,
                                                           scoring=metric)
            mean_score = np.mean(n_fold_score)
            self.model_scores[self.model].append({metric:mean_score})
        model_scores = self.model_scores
        return model_scores


