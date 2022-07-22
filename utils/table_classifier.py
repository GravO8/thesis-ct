import numpy as np
from utils.trainer import compute_metrics
from csv_loader import CSVLoader, SETS
from skopt import gp_minimize
from abc import ABC, abstractmethod

class TableClassifier(ABC):
    def __init__(self, loader: CSVLoader):
        self.loader = loader
        
    @abstractmethod
    def predict(self, x):
        pass
    
    def get_set(self, set):
        assert set, f"TableClassifier.get_set: Unknown set {set}. Available sets are {SETS}"
        set = self.loader.get_set(set)
        return set["x"], set["y"]
        
    def compute_metrics(self, set: str, verbose = False):
        x, y   = self.get_set(set) 
        y_prob = self.predict(x)
        if verbose:
            l, c = np.unique(y, return_counts = True)
            print(l, c)
        return compute_metrics(y, y_prob)
        
        
class ClassicClassifier(TableClassifier):
    def __init__(self, model, loader, ranges, metric = "f1_score", **kwargs):
        super().__init__(loader)
        self.model  = model
        self.metric = metric
        self.hyperparam_tune(**kwargs)
        
    def predict(self, x):
        try:
            return self.model.predict_proba(x)[:,1]
        except:
            return self.model.predict(x)
        
    def fit(self, **params):
        x_train, y_train = self.get_set("train")
        self.model.set_params(**params)
        self.model.fit(x_train, y_train)
        return self.compute_metrics("val")[self.metric]
        
    def hyperparam_tune(self, ranges, init_points = 5, n_iter = 20):
        res = gp_minimize(
            f = self.fit,
            dimensions = ranges,
            n_calls = n_iter,
            n_random_starts = init_points
        )
        self.params = res.x
        # optimizer = BayesianOptimization(
        #     # f       = lambda x: self.fit(x),
        #     f       = self.fit,
        #     pbounds = ranges,
        #     verbose = 2,
        #     random_state = 0,
        # )
        # optimizer.maximize(
        #     init_points = init_points,
        #     n_iter      = n_iter,
        # )
        # self.best_params = optimizer.max["params"]
        # self.fit(**self.best_params)
        
def knns(loader):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = ClassicClassifier(model   = KNN(), 
                            loader  = loader,
                            ranges = {
                                "n_neighbors": [3, 5, 7, 9, 11],
                                "metric": ["euclidean", "manhattan", "chebyshev", "jaccard"]
                            },)
    # print( knn.compute_metrics("test") )
    print( knn.params )
    
    
if __name__ == '__main__':
    from csv_loader import CSVLoader
    loader = CSVLoader("table_data.csv", "")
        
