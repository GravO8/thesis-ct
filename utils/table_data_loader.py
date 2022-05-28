import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

STAGE_BASELINE     = "baseline"
STAGE_PRETREATMENT = "pretreatment"
STAGE_24H          = "24h"
STAGE_DISCHARGE    = "discharge"
STAGES             = (STAGE_PRETREATMENT, STAGE_24H, STAGE_DISCHARGE)


def str_to_datetime(date: str):
    if (date is None) or (date == "None"):
        return None
    return datetime.fromisoformat(date[:19])
    
def get_age(birthdate, day):
    if (birthdate is None) or (day is None):
        return None
    return day.year - birthdate.year - ((day.month, day.day) < (birthdate.month, birthdate.day))
    
def get_hour_delta(time1, time2):
    if (time1 is None) or (time2 is None):
        return None
    return (time2 - time1).seconds // (60*60)
    
def convert_missing_to_nan(col):
    return [np.nan if (v == "None") or (v is None) else v for v in col]
    
def to_float(df: pd.DataFrame):
    for col in df.columns:
        df[col] = convert_missing_to_nan(df[col])
    return df.astype("float")


class TableDataLoader:
    def __init__(self, table_filename: str = "table_data.csv", 
    labels_filename: str = "dataset.csv", data_dir: str = None):
        data_dir        = "" if data_dir is None else data_dir
        table_filename  = os.path.join(data_dir, table_filename)
        labels_filename = os.path.join(data_dir, labels_filename)
        self.table_df   = pd.read_csv(table_filename)
        self.labels_df  = pd.read_csv(labels_filename)
        self.train_set  = {"x": None, "y": None}
        self.val_set    = {"x": None, "y": None}
        self.test_set   = {"x": None, "y": None}
        self.imputed    = False
        self.filter_no_ncct()
        self.add_vars()
        
    def filter_no_ncct(self):
        '''
        removes the rows from table_df whose patients don't have a NCCT exam
        '''
        ncct_ids      = self.labels_df["patient_id"].astype(str)
        self.table_df = self.table_df[self.table_df["idProcessoLocal"].isin(ncct_ids)]
        
    def add_vars(self):
        '''
        add the variables 'age' and 'time_since_onset'
        '''
        birthdate   = [str_to_datetime(date) for date in self.table_df["dataNascimento-1"].values]
        stroke_date = [str_to_datetime(date) for date in self.table_df["dataAVC-4"].values]
        ncct_time   = [str_to_datetime(date) for date in self.table_df["data-7"].values]
        age         = [get_age(birthdate[i],stroke_date[i]) for i in range(len(birthdate))]
        age         = [get_age(birthdate[i],ncct_time[i]) if age[i] is None else age[i] for i in range(len(birthdate))]
        onset_time  = [get_hour_delta(stroke_date[i],ncct_time[i]) for i in range(len(birthdate))]
        self.table_df["age"]              = age
        self.table_df["time_since_onset"] = onset_time
        
    def filter(self, to_remove: list = ["numRegistoGeral-1", "dataNascimento-1", 
        "rankin-2", "dataAVC-4", "data-7", "enfAntOutro-7", "ouTerrIsqOutro-7",
        "colaCTA1-8", "colaCTA2a-8", "colaCTA2b-8", "ocEst-9", "localiz-9", 
        "lado-9", "ocEst-10", "localiz-10", 
        "lado-10"], to_keep: list = None, stage: str = None):
        if stage is None:
            assert to_keep is not None
            self.filter_keep(to_keep)
        elif to_keep is None:
            assert stage is not None
            self.filter_remove(to_remove, stage)
    
    def filter_remove(self, to_remove: list, stage: str):
        to_remove.extend(["rankin-23", "NCCT", "CTA", "visible"])
        sections_to_remove = ["18"]
        if stage == STAGE_BASELINE:
            sections_to_remove.extend( ["7", "11", "15", "19", "20", "21", "22"] )
        elif stage == STAGE_PRETREATMENT:
            sections_to_remove.extend( ["11", "15", "19", "20", "21", "22"] )
        elif stage == STAGE_24H:
            sections_to_remove.extend( ["19", "20", "21", "22"] )
        elif stage != STAGE_DISCHARGE:
            assert False, f"TableDataLoader.filter: the available are {STAGES}"
        for col in self.table_df.columns:
            for section in sections_to_remove:
                if col.endswith(section):
                    to_remove.append(col)
        assert "idProcessoLocal" not in to_remove
        for col in set(to_remove):
            del self.table_df[col]
            
    def filter_keep(self, to_keep: list):
        assert "rankin-23" not in to_keep
        assert "idProcessoLocal" in to_keep
        for col in self.table_df.columns:
            if col not in to_keep:
                del self.table_df[col]
                
    def get_set(self, set: str):
        out        = {"x": None, "y": None}
        set_labels = self.labels_df[self.labels_df["set"] == set]
        set_ids    = set_labels["patient_id"].values.astype(str)
        out["x"]   = self.table_df[self.table_df["idProcessoLocal"].isin(set_ids)].copy()
        out["y"]   = set_labels["binary_rankin"]
        del out["x"]["idProcessoLocal"]
        return out
        
    def split(self):
        self.convert_boolean_sequences()
        self.train_set = self.get_set("train")
        self.val_set   = self.get_set("val")
        self.test_set  = self.get_set("test")
        del self.table_df
        
    def convert_boolean_sequences(self):
        boolean_cols = {"ouTerrIsq-7": 4, "ouTerrIsqL-7": 2, "lacAntL-7": 2, 
                        "enfAnt-7": 9, "enfAntL-7": 2, "compRtPA": 5, 
                        "TCCEter": 11, "TCCElac": 7, "RMNter": 11, "RMNlac": 7, 
                        "outProc": 4, "outCom": 8, "ecocarAnormal": 19}
        for col in boolean_cols:
            if col not in self.table_df.columns: continue
            splitted = [None if v == "None" else v.split(",") for v in self.table_df[col].values]
            new_cols = {f"{col}-{i+1}":[] for i in range(boolean_cols[col])}
            for s in splitted:
                if s is None:
                    for i in range(boolean_cols[col]):
                        new_cols[f"{col}-{i+1}"].append(None)
                else:
                    assert len(s) == boolean_cols[col]
                    for i in range(boolean_cols[col]):
                        new_cols[f"{col}-{i+1}"].append(int(s[i].lower() == "true"))
            del self.table_df[col]
            for col in new_cols:
                self.table_df[col] = new_cols[col]

    def impute(self):
        assert self.train_set["x"] is not None, "TableDataLoader.impute: call the 'split' method first"
        if self.imputed:
            return
        self.imputed        = True
        self.train_set["x"] = to_float(self.train_set["x"])
        self.val_set["x"]   = to_float(self.val_set["x"])
        self.test_set["x"]  = to_float(self.test_set["x"])
        imp = IterativeImputer(max_iter = 40, random_state = 0)
        imp.fit(self.train_set["x"])
        self.train_set["x"] = imp.transform(self.train_set["x"])
        self.val_set["x"]   = imp.transform(self.val_set["x"])
        self.test_set["x"]  = imp.transform(self.test_set["x"])
        
    def normalize(self):
        assert self.train_set["x"] is not None, "TableDataLoader.normalize: call the 'split' method first"
        assert self.imputed, "TableDataLoader.normalize: call the 'impute' method first"
        scaler = StandardScaler()
        scaler.fit(self.train_set["x"])
        self.train_set["x"] = scaler.transform(self.train_set["x"])
        self.val_set["x"]   = scaler.transform(self.val_set["x"])
        self.test_set["x"]  = scaler.transform(self.test_set["x"])
        print(self.train_set["x"])

if __name__ == "__main__":
    table_loader = TableDataLoader(data_dir = "../../../data/gravo/")
    table_loader.filter(stage = STAGE_PRETREATMENT)
    table_loader.split()
    table_loader.impute()
    table_loader.normalize()
