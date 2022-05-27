import os
import numpy as np
import pandas as pd
from datetime import datetime

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


class TableDataLoader:
    def __init__(self, table_filename: str = "table_data.csv", 
    labels_filename: str = "dataset.csv", data_dir: str = None):
        data_dir        = "" if data_dir is None else data_dir
        table_filename  = os.path.join(data_dir, table_filename)
        labels_filename = os.path.join(data_dir, labels_filename)
        self.table_df   = pd.read_csv(table_filename)
        self.labels_df  = pd.read_csv(labels_filename)
        self.train_set  = None
        self.val_set    = None
        self.test_set   = None
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
                
    def split(self):
        self.convert_boolean_sequences()
        self.amputate()
        train_ids = self.labels_df[self.labels_df["set"] == "train"]["patient_id"].values.astype(str)
        val_ids   = self.labels_df[self.labels_df["set"] == "val"]["patient_id"].values.astype(str)
        test_ids  = self.labels_df[self.labels_df["set"] == "test"]["patient_id"].values.astype(str)
        self.train_set = self.table_df[self.table_df["idProcessoLocal"].isin(train_ids)]
        self.val_set   = self.table_df[self.table_df["idProcessoLocal"].isin(val_ids)]
        self.test_set  = self.table_df[self.table_df["idProcessoLocal"].isin(test_ids)]
        del self.train_set["idProcessoLocal"]
        del self.val_set["idProcessoLocal"]
        del self.test_set["idProcessoLocal"]
        
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

    def amputate(self):
        for col in self.table_df.columns:
            self.table_df[col] = [np.nan if (v is None) or (v == "None")  else v for v in self.table_df[col]]
            self.table_df[col].astype("float")
        print(self.table_df.values)
        
    def normalize(self):
        assert self.train_set is not None, "TableDataLoader.normalize: call the 'split' method first"
        print(self.train_set.values)

if __name__ == "__main__":
    table_loader = TableDataLoader(data_dir = "../../../data/gravo/")
    table_loader.filter(stage = STAGE_PRETREATMENT)
    table_loader.split()
    # table_loader.normalize()
