import pandas as pd, os
from abc import ABC, abstractmethod

SETS = ("train", "val", "test")


class CSVLoader(ABC):
    def __init__(self, csv_filename: str, keep_cols: list, target_col: str, 
    normalize: bool, dirname = "", **kwargs):
        csv_filename = os.path.join(dirname, csv_filename)
        self.table   = pd.read_csv(csv_filename)
        keep_cols    = self.preprocess(keep_cols, **kwargs)
        self.set_sets(keep_cols, target_col)
        
    def set_sets(self, keep_cols: list, target_col: str, set_col: str, 
    normalize: bool):
        self.sets = {}
        self.split(set_col)
        self.filter(keep_cols, target_col)
        self.normalize()
        
    @abstractmethod
    def preprocess(self):
        pass
        
    def split(self, set_col: str = "set"):
        assert self.table is not None
        for s in SETS:
            self.sets[s] = self.table[self.table[set_col] == s].copy()
        self.table = None
        
    def filter(self, remove_cols: list, target_col: str):
        assert self.table is None, "CSVLoader.filter: call split first"
        keep_cols = [col for col in self.table.columns if col not in remove_cols]
        for s in SETS:
            x = self.sets[s][keep_cols]
            y = self.sets[s][target_col].values
            self.sets[s] = {"x": x, "y": y}
            
    def get_set(self, set: str):
        assert set in SETS
        return self.sets[s]
        
    def normalize(self):
        scaler = StandardScaler()
        scaler.fit(self.sets["train"]["x"])
        for s in SETS:
            self.sets[s]["x"] = scaler.transform(self.sets[s]["x"])
