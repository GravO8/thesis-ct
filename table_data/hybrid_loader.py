import pandas as pd
from table_loader import TableLoader
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class HybridLoader(TableLoader):
    def __init__(self, *args, loader_complementar = None, **kwargs):
        self.loader_complementar = loader_complementar
        super().__init__(*args, **kwargs)
    
    def stage_cols(self, stage):
        assert stage == "baseline"
        return super().stage_cols(stage) + [col for col in list(self.table.columns) if col.startswith("feature")]
        
    def feature_selection(self):
        N        = 64
        pca      = PCA(n_components = N)
        cols     = [col for col in self.sets["train"]["x"].columns if col.startswith("feature")]
        cols_pca = [f"pca{i}" for i in range(N)]
        pca.fit(self.sets["train"]["x"][cols])
        for set in self.available_sets():
            transformed = pca.transform(self.sets[set]["x"][cols])
            self.sets[set]["x"] = self.sets[set]["x"].drop(cols, axis = 1)
            self.sets[set]["x"][cols_pca] = transformed

    def impute(self):
        imp  = IterativeImputer(max_iter = 40, random_state = 0)
        cols = super().stage_cols("baseline")
        if self.loader_complementar is None: 
            imp.fit(self.sets["train"]["x"][cols])
        else:
            imp.fit(self.loader_complementar.sets["train"]["x"])
        for s in self.available_sets():
            self.sets[s]["x"][cols] = imp.transform(self.sets[s]["x"][cols])
