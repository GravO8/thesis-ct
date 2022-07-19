import os, numpy as np, pandas as pd, sys
# sys.path.append("..")
from sklearn.preprocessing import StandardScaler
from ..utils.csv_loader import CSVLoader

MIL = "mil" # multiple instance learning
SIL = "sil" # single instance learning
REGIONS = ("caudate", "insula", "internal_capsule", "lentiform", "m1", "m2", "m3", "m4", "m5", "m6")

class RadiomicsLoader(CSVLoader):
    def __init__(self, table_filename, data_dir: str = None, target: str = "aspects"):
        assert target in ("aspects","binary_aspects","binary_rankin","rankin")
        data_dir        = "" if data_dir is None else data_dir
        table_filename  = os.path.join(data_dir, table_filename)
        self.table_df   = pd.read_csv(table_filename)
        self.sets       = {}
        self.target     = target
        self.set_name   = "aspects_set" if "aspects" in self.target else "rankin_set"
        self.mode       = None
        
    def filter(self, mode = MIL):
        assert mode in (MIL, SIL)
        self.mode = mode
        if mode == SIL:
            self.table_df = self.table_df[self.table_df["Mask"] == "MNI152_T1_2mm_brain_mask"].copy()
        else:
            self.table_df = self.table_df[self.table_df["Mask"] != "MNI152_T1_2mm_brain_mask"].copy()
        vars          = [v for v in self.table_df.columns if v.startswith("original") or (v in (self.target, self.set_name, "Mask", "Image"))]
        self.table_df = self.table_df[vars].copy()
        self.set_columns()
        
    def set_columns(self):
        self.columns = list(self.table_df.columns)
        del self.columns[self.columns.index("Mask")]
        del self.columns[self.columns.index("Image")]
        del self.columns[self.columns.index(self.target)]
        del self.columns[self.columns.index(self.set_name)]
        
    def set_set(self, set: str):
        assert self.mode is not None, "TableDataLoader.get_set: call the 'filter' method first"
        return self.table_df[self.table_df[self.set_name] == set].copy()
    
    def get_set(self, set: str):
        assert set in ("train", "val", "test")
        out = {}
        if self.mode == SIL:
            out["x"] = self.sets[set][self.columns]
            out["y"] = self.sets[set][self.target]
        else:
            out["x"], out["y"] = [], []
            for patient in np.unique( self.sets[set]["Image"].values ):
                rows = self.sets[set][self.sets[set]["Image"] == patient]
                instances = []
                for region in REGIONS:
                    instances.append( rows[rows["Mask"] == f"{region}_L_2mm"][self.columns] )
                    instances.append( rows[rows["Mask"] == f"{region}_R_2mm"][self.columns] )
                out["x"].append(instances)
                out["y"].append(rows[self.target].values[0])
        return out
        
    def normalize(self):
        assert len(self.sets) == 3, "TableDataLoader.normalize: call the 'split' method first"
        if self.mode == SIL:
            scaler = StandardScaler()
            scaler.fit(self.sets["train"][self.columns])
            for set in ("train", "val", "test"):
                self.sets[set][self.columns] = scaler.transform(self.sets[set][self.columns])
        else:
            for region in REGIONS:
                for region in (f"{region}_L_2mm", f"{region}_R_2mm"):
                    scaler = StandardScaler()
                    scaler.fit(self.sets["train"][self.sets["train"]["Mask"] == region][self.columns])
                    for set in ("train", "val", "test"):
                        self.sets[set].loc[self.sets[set]["Mask"] == region, self.columns] = scaler.transform(self.sets[set][self.sets[set]["Mask"] == region][self.columns])

    def amputate(self):
        assert False
    def impute(self):
        assert False


if __name__ == "__main__":
    loader = RadiomicsLoader(table_filename = "ncct_radiomic_features.csv", data_dir = "radiomic_features")
    loader.filter("mil")
    loader.split()
    loader.normalize()
