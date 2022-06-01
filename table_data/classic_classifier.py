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
        self.sets["train"] = table_loader.train_set
        self.sets["val"]   = table_loader.val_set
        self.sets["test"]  = table_loader.test_set
        self.columns       = list(table_loader.train_set.columns)
        del self.columns[self.columns.index("binary_rankin")]
        del self.columns[self.columns.index("idProcessoLocal")]
        print(len(table_loader.train_set), len(table_loader.val_set), len(table_loader.test_set))
        
    def get_columns(self):
        return self.columns
        


def knn(loader):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = ClassicClassifier(model = KNN(), 
                            param_grid = {
                                "n_neighbors": [3, 5, 7, 9, 11],
                                "metric": ["euclidean", "manhattan", "chebyshev", "jaccard"]
                            },
                            table_loader = loader)
    print( knn.compute_metrics("test") )
    print( knn.params )
    
    
# def decision_trees():
#     from sklearn.tree import DecisionTreeClassifier as DT
#     dt = ClassicClassifier(model = DT(),
#                            param_grid = {
#                                 "": ,
# 
#                            })
#     print( dt.compute_metrics("test") )
#     print( dt.params )


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from utils.table_data_loader import TableDataLoader
    
    loader = TableDataLoader(data_dir = "../../../data/gravo/")
    knn(loader)
