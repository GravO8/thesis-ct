import numpy as np, skopt
from .csv_loader import CSVLoader, SETS
from .trainer import compute_metrics
from abc import ABC, abstractmethod


class TableClassifier(ABC):
    def __init__(self, loader: CSVLoader):
        self.loader = loader
        
    @abstractmethod
    def predict(self, x):
        pass
    
    def get_set(self, set: str):
        set = self.loader.get_set(set)
        return set["x"], set["y"]
        
    def compute_metrics(self, set: str, verbose = True):
        x, y   = self.get_set(set) 
        y_prob = self.predict(x)
        if verbose:
            l, c = np.unique(y, return_counts = True)
            print(l, c)
        return compute_metrics(y, y_prob)
        
    def plot_2d(self, set: str):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        x, y        = self.get_set(set)
        transformer = TSNE(n_components = 2)
        x_2d        = transformer.fit_transform(x)
        colors      = ["red" if el else "blue" for el in y]
        plt.scatter(x_2d[:,0], x_2d[:,1], c = colors)
        plt.show()
        
        
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
        
    def hyperparam_tune(self, ranges, init_points = 5, n_iter = 20, cv = 5, scoring = "f1", verbose = True):
        opt = skopt.BayesSearchCV(
            self.model,
            ranges,
            n_iter  = n_iter,
            cv      = cv,
            scoring = scoring,
            verbose = 0,
            return_train_score = True)
        x_train, y_train = self.get_set("train")
        opt.fit(x_train, y_train)
        # import pandas as pd
        # pd.DataFrame(opt.cv_results_).to_csv("sapo.csv", index = False)
        self.best_params = opt.best_params_
        self.fit()
        if verbose:
            print("val score:", opt.best_score_)
            
    def grid_hyperparam_tune(self, ranges):
        from sklearn.model_selection import ParameterGrid
        from copy import deepcopy
        x_train, y_train = self.get_set("train")
        best_score = None
        best_model = None
        best_param = None
        for params in ParameterGrid(ranges):
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            metrics = self.compute_metrics("val")
            if (best_score is None) or (metrics["f1-score"] > best_score):
                best_score = metrics["f1-score"]
                best_model = deepcopy(self.model)
                best_param = params
        self.model = best_model
        self.best_params = best_param


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
    print("train", knn.compute_metrics("train"))
    print("test", knn.compute_metrics("test"))
    print( knn.best_params )


def decision_trees(loader, **kwargs):
    from sklearn.tree import DecisionTreeClassifier as DT
    dt = ClassicClassifier( model    = DT(),
                            loader   = loader,
                            ranges   = {
                                "criterion": ["entropy", "gini"],
                                "max_depth": skopt.space.space.Integer(2, 25),
                                "min_impurity_decrease": skopt.space.space.Real(.001, .025),
                                "random_state": [0]
                           },
                           # ranges = {
                           #      "criterion": ["entropy", "gini"],
                           #      "max_depth": [2, 5, 10, 15, 20, 25],
                           #      "min_impurity_decrease": [0.025, 0.01, 0.005, 0.0025, 0.001],
                           #      "random_state": [0]
                           # },
                           **kwargs)
    print("train", dt.compute_metrics("train", verbose = True))
    print("test", dt.compute_metrics("test", verbose = True))
    print( dt.best_params )


def random_forests(loader, **kwargs):
    from sklearn.ensemble import RandomForestClassifier as RF
    rf = ClassicClassifier( model   = RF(),
                            loader  = loader,
                            ranges  = {
                                "criterion": ["entropy", "gini"],
                                "max_depth": skopt.space.space.Integer(2, 25),
                                "n_estimators": skopt.space.space.Integer(5, 300),
                                "max_features": skopt.space.space.Real(.1, 1),
                                "random_state": [0]
                           },
                           **kwargs)
    print("train", rf.compute_metrics("train"))
    print("test", rf.compute_metrics("test"))
    print( rf.best_params )
    
    
def logistic_regression(loader, **kwargs):
    from sklearn.linear_model import LogisticRegression as LR
    lr = ClassicClassifier( model   = LR(),
                            loader  = loader,
                            ranges  = {
                                "C": skopt.space.space.Real(.001, 3),
                                "tol": skopt.space.space.Real(1e-5, 1e-1),
                                "penalty": ["l1", "l2"],
                                "solver": ["saga"],
                                "random_state": [0],
                                "max_iter": [10000],
                           },
                           **kwargs)
    print("train", lr.compute_metrics("train"))
    print("test", lr.compute_metrics("test"))
    print( lr.best_params )
    
    
def gradient_boosting(loader, **kwargs):
    from sklearn.ensemble import GradientBoostingClassifier as GB
    gb = ClassicClassifier( model   = GB(),
                            loader  = loader,
                            ranges  = {
                                "n_estimators": skopt.space.space.Integer(5, 300),
                                "max_depth": skopt.space.space.Integer(5, 25),
                                "learning_rate": skopt.space.space.Real(.1, .9),
                                "random_state": [0],
                           },
                           **kwargs)
    print("train", gb.compute_metrics("train"))
    print("test", gb.compute_metrics("test"))
    print( gb.best_params )
