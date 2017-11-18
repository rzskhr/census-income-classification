print("Hello, World!")

model_selection = 0

class ModelMetrics(object):

    # Random permutation cross-validator with 80-20 train test split
    cv = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.2)
    # dictionary to store scores
    model_scores = defaultdict(list)

    def __init__(self, model_name, model_obj, features, response):
        self.model_name = model_name
        self.model_obj = model_obj
        ModelMetrics.model_scores[model_name] = list()
        self.cv = ModelMetrics.cv
        self.features = features
        self.response = response

    def model_scoring(self, scoring_metric):
        for metric in scoring_metric:
            n_fold_score = cross_val_score(self.model_obj, self.features,
                                                           self.response,
                                                           cv=self.cv,
                                                           scoring=metric)
            model_scores[self.model_name].append({metric:np.mean(n_fold_score)})



