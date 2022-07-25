import numpy as np
from .csv_loader import CSVLoader, SETS
from .trainer import compute_metrics
from skopt import BayesSearchCV
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
        self.hyperparam_tune(ranges, **kwargs)
        
    def predict(self, x):
        try:
            return self.model.predict_proba(x)[:,1]
        except:
            return self.model.predict(x)
        
    def fit(self, params = None):
        x_train, y_train = self.get_set("train")
        self.model.set_params(**(self.best_params if params is None else params))
        self.model.fit(x_train, y_train)
        
    def hyperparam_tune(self, ranges, init_points = 5, n_iter = 20, cv = 5):
        opt = BayesSearchCV(
            self.model,
            ranges,
            n_iter = n_iter,
            cv = cv)
        x_train, y_train = self.get_set("train")
        opt.fit(x_train, y_train)
        self.best_params = opt.best_params_
        self.fit()
        
def knns(loader, **kwargs):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = ClassicClassifier(model   = KNN(), 
                            loader  = loader,
                            ranges  = {
                                "n_neighbors": [3, 5, 7, 9, 11],
                                "weights": ["uniform", "distance"],
                                "metric": ["euclidean", "manhattan", "chebyshev", "jaccard"]
                            },
                            **kwargs)
    print( knn.compute_metrics("test") )
    print( knn.best_params )
        
