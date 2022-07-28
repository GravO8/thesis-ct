import pandas as pd, os

REGIONS = {"m1":"M1", "m2":"M2", "m3":"M3", "m4":"M4", "m5":"M5", "m6":"M6", "caudate":"C", "insula":"I", "internal_capsule":"IC", "lentiform":"L"}
SETS    = ["train", "val", "test"]
ALL     = "all"
IMAGE   = "Image"
MASK    = "Mask"


class ASPECTSMILLoader:
    def __init__(self, csv_filename: str, keep_cols: list, binary: bool = False, 
    normalize: bool = True, dirname: str = ""):
        csv_filename = os.path.join(dirname, csv_filename)
        self.table   = pd.read_csv(csv_filename)
        self.set_sets(keep_cols, binary)
        
    def set_sets(self, keep_cols, binary: bool):
        self.sets   = {}
        keep_cols   = self.all_cols() if keep_cols == ALL else keep_cols
        target_col  = "binary_aspects" if binary else "aspects"
        set_col     = "aspects_set"
        for s in SETS:
            self.sets[s] = {"x": [], "y": [], "instance_labels": []}
            patients     = self.table[self.table[set_col] == s][IMAGE].unique()
            for patient in patients:
                patient_features = []
                rows             = self.table[self.table[IMAGE] == patient]
                for region in REGIONS:
                    patient_features.append( rows[rows[MASK] == f"{region}_L_2mm"][keep_cols].values )
                    patient_features.append( rows[rows[MASK] == f"{region}_R_2mm"][keep_cols].values )
                y = rows[target_col].unique()
                assert len(y) == 1
                assert len(patient_features) == 20
                self.sets[s]["x"].append( patient_features )
                self.sets[s]["y"].append( y[0] )
                self.sets[s]["instance_labels"].append( {REGIONS[r]: rows[REGIONS[r]].values[0].astype(int) for r in REGIONS} )
        
    def all_cols(self):
        return [col for col in self.table.columns if col.startswith("original_")]
    
    def available_sets(self):
        return [s for s in self.sets]
        
    def get_set(self, set: str):
        assert set in self.available_sets(), f"ASPECTSMILLoader.get_set: Unknown set {set}. Available sets are {self.loader.available_sets()}"
        return self.sets[set]

if __name__ == "__main__":
    loader = ASPECTSMILLoader("ncct_radiomic_features.csv", 
                            "all", dirname = "../../../data/gravo")
