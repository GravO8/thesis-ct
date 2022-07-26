import pandas as pd, os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from abc import ABC, abstractmethod

SETS = ("train", "val", "test")

def convert_missing_to_nan(col):
    return np.array([np.nan if (v == "None") or (v is None) else v for v in col]).astype("float")


class CSVLoader(ABC):
    def __init__(self, csv_filename: str, keep_cols: list, target_col: str, 
    set_col: str = "set", normalize: bool = True, 
    empty_values_method: str = "amputate", join_train_val = False, dirname = "", **kwargs):
        csv_filename = os.path.join(dirname, csv_filename)
        self.table   = pd.read_csv(csv_filename)
        keep_cols    = self.preprocess(keep_cols, **kwargs)
        self.set_sets(keep_cols, target_col, set_col, normalize, empty_values_method, join_train_val)
        
    def set_sets(self, keep_cols: list, target_col: str, set_col: str, 
    normalize: bool, empty_values_method: str, join_train_val: bool):
        self.sets = {}
        self.split(set_col, join_train_val)
        self.filter(keep_cols, target_col)
        self.to_float()
        self.empty_values(empty_values_method)
        if normalize:
            self.normalize()
            
    def empty_values(self, method):
        if method == "amputate":
            self.amputate()
        elif method == "impute":
            self.impute()
        else:
            assert False
        
    @abstractmethod
    def preprocess(self):
        pass
        
    def split(self, set_col: str = "set", join_train_val: bool = False):
        global SETS
        assert self.table is not None
        if join_train_val:
            self.table.loc[self.table[set_col] == "val", set_col] = "train"
            SETS = ["train", "test"]
        for s in SETS:
            self.sets[s] = self.table[self.table[set_col] == s].copy()
        self.table = None
        
    def filter(self, keep_cols: list, target_col: str):
        assert self.table is None, "CSVLoader.filter: call split first"
        for s in SETS:
            x = self.sets[s][keep_cols]
            y = self.sets[s][target_col].values
            self.sets[s] = {"x": x, "y": y}
            
    def get_set(self, set: str):
        assert set in SETS
        return self.sets[set]
        
    def normalize(self):
        scaler = StandardScaler()
        scaler.fit(self.sets["train"]["x"])
        for s in SETS:
            self.sets[s]["x"] = scaler.transform(self.sets[s]["x"])

    def impute(self):
        imp = IterativeImputer(max_iter = 40, random_state = 0)
        imp.fit(self.sets["train"]["x"])
        for s in SETS:
            self.sets[s]["x"] = imp.transform(self.sets[s]["x"])
        
    def amputate(self):
        for s in SETS:
            to_keep = []
            for _, row in self.sets[s]["x"].iterrows():
                if row.isnull().values.any():
                    to_keep.append(False)
                else:
                    to_keep.append(True)
            self.sets[s]["x"] = self.sets[s]["x"].iloc[to_keep]
            self.sets[s]["y"] = self.sets[s]["y"][to_keep]
            
    def to_float(self):
        for s in SETS:
            for col in self.sets[s]["x"].columns:
                self.sets[s]["x"][col] = convert_missing_to_nan(self.sets[s]["x"][col])
