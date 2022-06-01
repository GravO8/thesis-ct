import sys
sys.path.append("..")
from sklearn.model_selection import ParameterGrid
from table_classifier import TableClassifier
from copy import deepcopy

class ClassicClassifier(TableClassifier):
    def __init__(self, model, param_grid: dict, table_loader, stage: str = "baseline"):
        super().__init__(table_loader, stage = stage)
        self.model  = model
        self.params = None
        self.fit(param_grid)
        
    def get_predictions(self, x):
        try:
            return self.model.predict_proba(x)[:,1]
        except:
            return self.model.predict(x)
        
    def fit(self, param_grid: dict):
        x_train    = self.get_x("train")
        y_train    = self.get_y("train")
        best_score = None
        best_model = None
        best_param = None
        for params in ParameterGrid(param_grid):
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            metrics = self.compute_metrics("val")
            if (best_score is None) or (metrics["f1-score"] > best_score):
                best_score = metrics["f1-score"]
                best_model = deepcopy(self.model)
                best_param = params
        self.model = best_model
        self.params = best_param
    
    def load_sets(self, table_loader, stage: str):
        table_loader.filter(stage = stage)
        table_loader.split()
        table_loader.amputate()
        # table_loader.impute()
        self.sets["train"] = table_loader.train_set
        self.sets["val"]   = table_loader.val_set
        self.sets["test"]  = table_loader.test_set
        self.columns       = list(table_loader.train_set.columns)
        del self.columns[self.columns.index("binary_rankin")]
        del self.columns[self.columns.index("idProcessoLocal")]
        print(len(table_loader.train_set), len(table_loader.val_set), len(table_loader.test_set))
        
    def get_columns(self):
        return self.columns
        


def knns(loader):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = ClassicClassifier(model = KNN(), 
                            param_grid = {
                                "n_neighbors": [3, 5, 7, 9, 11],
                                "metric": ["euclidean", "manhattan", "chebyshev", "jaccard"]
                            },
                            table_loader = loader)
    print( knn.compute_metrics("test") )
    print( knn.params )
    
    
def decision_trees(loader):
    from sklearn.tree import DecisionTreeClassifier as DT
    dt = ClassicClassifier(model = DT(),
                           param_grid = {
                                "criterion": ["entropy", "gini"],
                                "max_depth": [2, 5, 10, 15, 20, 25],
                                "min_impurity_decrease": [0.025, 0.01, 0.005, 0.0025, 0.001],
                                "random_state": [0]
                           },
                           table_loader = loader)
    print( dt.compute_metrics("test") )
    print( dt.params )
    
    
def random_forests(loader):
    from sklearn.ensemble import RandomForestClassifier as RF
    rf = ClassicClassifier(model = RF(),
                           param_grid = {
                                "criterion": ["entropy", "gini"],
                                "max_depth": [5, 10, 20, 25],
                                "n_estimators": [5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
                                "max_features": [.1, .3, .5, .7, .9, 1],
                                "random_state": [0]
                           },
                           table_loader = loader)
    print( rf.compute_metrics("test") )
    print( rf.params )


def logistic_regression(loader):
    from sklearn.linear_model import LogisticRegression as LR
    lr = ClassicClassifier(model = LR(),
                           param_grid = {
                                "C": [0.01, 0.1, 0.2, 0.5, 1, 2, 3],
                                "tol": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                                "penalty": ["l1", "l2"],
                                "solver": ["saga"],
                                "random_state": [0],
                           },
                           table_loader = loader)
    print( lr.compute_metrics("test") )
    print( lr.params )
    
    
def gradient_boosting(loader):
    from sklearn.ensemble import GradientBoostingClassifier as GB
    gb = ClassicClassifier(model = GB(),
                           param_grid = {
                                "n_estimators": [5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
                                "max_depth": [5, 10, 20, 25],
                                "learning_rate": [.1, .3, .5, .7, .9],
                                "random_state": [0],
                           },
                           table_loader = loader)
    print( gb.compute_metrics("test") )
    print( gb.params )
    

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils.table_data_loader import TableDataLoader
    
    loader = TableDataLoader(data_dir = "../../../data/gravo/")
    gradient_boosting(loader)
