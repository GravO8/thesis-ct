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
    empty_values_method: str = "amputate", join_train_val: bool = False, 
    dirname: str = "", reshuffle: int = False, **kwargs):
        csv_filename = os.path.join(dirname, csv_filename)
        self.table   = pd.read_csv(csv_filename)
        keep_cols    = self.preprocess(keep_cols, **kwargs)
        self.set_sets(keep_cols, target_col, set_col, normalize, empty_values_method, join_train_val, reshuffle)
        
    def set_sets(self, keep_cols: list, target_col: str, set_col: str, 
    normalize: bool, empty_values_method: str, join_train_val: bool, reshuffle: bool):
        self.sets = {}
        self.split(set_col, join_train_val, reshuffle, target_col)
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
        
    def reshuffle(self, set_col: str, target_col: str):
        distr = {}
        labels = None
        for s in SETS:
            labels, distr[s] = np.unique(self.table[self.table[set_col] == s][target_col].values, return_counts = True)
        self.table  = self.table.sample(frac = 1).reset_index(drop = True)
        self.sets_from_distr(set_col, labels, distr, target_col)
        
    def sets_from_distr(self, set_col: str, labels: list, distr: dict, target_col: str):
        set_col_vals = []
        sets         = {}
        for i in range(len(labels)):
            label = labels[i]
            if np.isnan(label): continue
            sets[label] = []
            for s in distr:
                sets[label].extend( [s] * distr[s][i] )
        for _, row in self.table.iterrows():
            label = row[target_col]
            if np.isnan(label):
                set_col_vals.append(np.nan)
            else:
                if len(sets[label]) == 0:
                    # set_col_vals.append( np.random.choice(SETS) )
                    set_col_vals.append(np.nan)
                else:
                    set_col_vals.append(sets[label].pop(0))
        self.table[set_col] = set_col_vals
        # self.table.to_csv("sapo.csv", index = False)
        # for s in distr:
        #     for i in range(len(labels)):
        #         label       = labels[i]
        #         c           = labels_count[label]
        #         label_rows  = self.table[self.table[target_col] == label]
        #         to_add      = distr[s][i]
        #         if s in self.sets:
        #             self.sets[s] = pd.concat([self.sets[s], label_rows.iloc[c:c+to_add].copy()])
        #         else:
        #             self.sets[s] = label_rows.iloc[c:c+to_add].copy()
        #         labels_count[label] += to_add
                
    def add_all_col(self, target_col: str, sets_distr: dict = {"train": 0.785, "val": 0.085, "test": 0.13}):
        if "all_set" in self.table.columns:
            return
        labels, total_distr = np.unique(self.table[target_col].values, return_counts = True)
        distr = {s: [] for s in sets_distr}
        for s in sets_distr:
            for i in range(len(labels)):
                distr[s].append( int(sets_distr[s] * total_distr[i]) )
        self.sets_from_distr("all", labels, distr, target_col)
        
    def split(self, set_col: str, join_train_val: bool, reshuffle: bool, target_col: str):
        global SETS
        assert self.table is not None
        if set_col == "all":
            self.add_all_col(target_col)
        if join_train_val:
            self.table.loc[self.table[set_col] == "val", set_col] = "train"
            SETS = ["train", "test"]
        if reshuffle:
            self.reshuffle(set_col, target_col)
        for s in SETS:
            self.sets[s] = self.table[self.table[set_col] == s].copy()
        self.table = None
        
    def filter(self, keep_cols: list, target_col: str):
        assert self.table is None, "CSVLoader.filter: call split first"
        for s in SETS:
            x = self.sets[s][keep_cols]
            y = self.sets[s][target_col].values
            self.sets[s] = {"x": x, "y": y}
            print(s, np.unique(y, return_counts = True)[1])
            
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
