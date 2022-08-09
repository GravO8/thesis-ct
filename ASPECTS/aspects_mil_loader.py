import pandas as pd, os, torch, numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

REGIONS = {"m1":"M1", "m2":"M2", "m3":"M3", "m4":"M4", "m5":"M5", "m6":"M6", "caudate":"C", "insula":"I", "internal_capsule":"IC", "lentiform":"L"}
SETS    = ["train", "val", "test"]
ALL     = "all"
LAWS    = "laws"
IMAGE   = "Image"
MASK    = "Mask"
RADIOMICS = "radiomics"


class ASPECTSMILLoader:
    def __init__(self, csv_filename: str, keep_cols: list, binary: bool = False, 
    normalize: bool = True, dirname: str = "", set_col: str = "aspects_set", 
    feature_selection: bool = False):
        csv_filename = os.path.join(dirname, csv_filename)
        self.table   = pd.read_csv(csv_filename)
        self.set_sets(keep_cols, binary, set_col)
        if feature_selection:
            self.feature_selection()
        if normalize:
            self.normalize()
        self.to_tensor()
        
    def to_tensor(self):
        for s in self.available_sets():
            self.sets[s]["x"] = torch.Tensor(self.sets[s]["x"])
            self.sets[s]["y"] = torch.Tensor(self.sets[s]["y"])
            
    def normalize(self):
        shape = self.sets["train"]["x"].shape # (#samples, #instances = #regions, #features)
        for region in range(shape[1]):
            scaler = StandardScaler()
            col    = self.sets["train"]["x"][:,region,:]
            scaler.fit(col)
            for s in self.available_sets():
                self.sets[s]["x"][:,region,:] = scaler.transform(self.sets[s]["x"][:,region,:])
                
    def feature_selection(self, select_N: int = 22):
        shape    = self.sets["train"]["x"].shape # (#samples, #instances = #regions, #features)
        new_sets = {s:None for s in self.available_sets()}
        for region in range(0,shape[1],2):
            col       = list(self.sets["train"]["x"][:,region,:]) + list(self.sets["train"]["x"][:,region+1,:])
            variances = np.var(col, axis = 0)
            cols_keep = np.argsort(variances)[::-1][:select_N]
            for s in self.available_sets():
                if new_sets[s] is None:
                    new_sets[s] = self.sets[s]["x"][:,region:region+2,cols_keep]
                else:
                    new_sets[s] = np.concatenate((new_sets[s], self.sets[s]["x"][:,region:region+2,cols_keep]), axis = 1 )
        for s in self.available_sets():
            self.sets[s]["x"] = np.array( new_sets[s] )    
        
    def set_sets(self, keep_cols, binary: bool, set_col: str, verbose = True):
        self.sets   = {}
        if keep_cols == ALL:
            keep_cols = self.all_cols()
        elif keep_cols == LAWS:
            keep_cols = self.law_cols()
        elif keep_cols == RADIOMICS:
            keep_cols = self.radiomic_cols()
        target_col  = "binary_aspects" if binary else "aspects"
        for s in SETS:
            patients     = self.table[self.table[set_col] == s][IMAGE].unique()
            if len(patients) == 0: 
                continue
            self.sets[s] = {"x": [], "y": [], "instance_labels": [], "patients": []}
            for patient in patients:
                patient_features = []
                rows             = self.table[self.table[IMAGE] == patient]
                for region in REGIONS:
                    patient_features.append( rows[rows[MASK] == f"{region}_L_2mm"][keep_cols].values[0] )
                    patient_features.append( rows[rows[MASK] == f"{region}_R_2mm"][keep_cols].values[0] )
                y       = rows[target_col].unique()
                patient = rows["Image"].unique()
                assert len(y) == 1
                assert len(patient) == 1
                assert len(patient_features) == 20
                if np.isnan(y[0]):
                    continue
                self.sets[s]["x"].append( patient_features )
                self.sets[s]["y"].append( y[0] )
                self.sets[s]["instance_labels"].append( {REGIONS[r]: rows[REGIONS[r]].values[0].astype(int) for r in REGIONS} )
                self.sets[s]["patients"].append( patient[0] )
            self.sets[s]["x"] = np.array(self.sets[s]["x"])
            self.sets[s]["y"] = np.array(self.sets[s]["y"])
            if verbose:
                print(s, self.sets[s]["x"].shape, self.sets[s]["y"].shape)
        
    def all_cols(self):
        return [col for col in self.table.columns if col.startswith("original_")]
        
    def law_cols(self):
        return [col for col in self.table.columns if "laws" in col]
        
    def radiomic_cols(self):
        return [col for col in self.table.columns if ("laws" not in col) and col.startswith("original_")]
    
    def available_sets(self):
        return [s for s in self.sets]
        
    def get_set(self, set: str):
        assert set in self.available_sets(), f"ASPECTSMILLoader.get_set: Unknown set {set}. Available sets are {self.loader.available_sets()}"
        for i in range(len(self.sets[set]["x"])):
            yield self.sets[set]["x"][i], self.sets[set]["y"][i]
            
    def get_test_instance_labels(self):
        return self.sets["test"]["instance_labels"]
            
    def len(self, set: str):
        assert set in self.available_sets(), f"ASPECTSMILLoader.len: Unknown set {set}. Available sets are {self.loader.available_sets()}"
        return len(self.sets[set]["y"])
        
    def get_patients(self, set: str):
        assert set in self.available_sets(), f"ASPECTSMILLoader.get_patients: Unknown set {set}. Available sets are {self.loader.available_sets()}"
        return self.sets[set]["patients"]
            
class ASPECTSMILLoaderBebug(ASPECTSMILLoader):
    N = 2
    def set_sets(self, keep_cols, binary: bool, verbose = True):
        self.sets   = {}
        keep_cols   = self.all_cols() if keep_cols == ALL else keep_cols
        target_col  = "binary_aspects" if binary else "aspects"
        set_col     = "aspects_set"
        for s in SETS:
            self.sets[s] = {"x": [], "y": []}
            patients     = self.table[self.table[set_col] == s][IMAGE].unique()
            for patient in patients:
                patient_features = []
                rows             = self.table[self.table[IMAGE] == patient]
                y                = rows[target_col].unique()
                assert len(y) == 1
                y = y[0].astype(int)
                positive = np.random.choice([0,1,2,3,4,5,6,7,8,9], 10-y, replace = False)
                for i in range(10):
                    if i in positive:
                        patient_features.append( [1]*ASPECTSMILLoaderBebug.N )
                    else:
                        patient_features.append( [0]*ASPECTSMILLoaderBebug.N )
                    patient_features.append( [0]*ASPECTSMILLoaderBebug.N )
                self.sets[s]["x"].append( patient_features )
                self.sets[s]["y"].append( y )
            self.sets[s]["x"] = np.array(self.sets[s]["x"])
            self.sets[s]["y"] = np.array(self.sets[s]["y"])
            if verbose:
                print(s, self.sets[s]["x"].shape, self.sets[s]["y"].shape)
    

if __name__ == "__main__":
    loader = ASPECTSMILLoader("ncct_radiomic_features.csv", 
                            ALL, dirname = "../../../data/gravo")
