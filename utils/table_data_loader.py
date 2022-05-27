import os
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
        "colaCTA1-8", "colaCTA2a-8", 
        "colaCTA2b-8", "ocEst-9", "localiz-9", "lado-9", "ocEst-10", "localiz-10", 
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
        train_ids = self.labels_df[self.labels_df["set"] == "train"]["patient_id"].values.astype(str)
        val_ids   = self.labels_df[self.labels_df["set"] == "val"]["patient_id"].values.astype(str)
        test_ids  = self.labels_df[self.labels_df["set"] == "test"]["patient_id"].values.astype(str)
        self.train_set = self.table_df[self.table_df["idProcessoLocal"].isin(train_ids)]
        self.val_set   = self.table_df[self.table_df["idProcessoLocal"].isin(val_ids)]
        self.test_set  = self.table_df[self.table_df["idProcessoLocal"].isin(test_ids)]
        del self.train_set["idProcessoLocal"]
        del self.val_set["idProcessoLocal"]
        del self.test_set["idProcessoLocal"]
        self.remove_rows()
        
    def missing_percentage(self, row):
        '''
        There are more patients with a lot of missing columns than columns with
        missing patients. Thus, it is better to remove these patients with lots
        of missing data
        '''
        missing = 0
        for col in self.train_set.columns:
            if (row[col] is None) or (row[col] == "None"):
                missing += 1
                print(col)
        print("---------------------")
        return missing/len(self.train_set.columns)
        
    def remove_rows(self):
        missing_rows = [self.missing_percentage(row) for _, row in self.table_df.iterrows()]
        d = {}
        for m in missing_rows:
            if m in d: d[m] += 1
            else: d[m] = 1
        print( dict(sorted(d.items())) ) 
        
        
        
    def normalize(self):
        assert self.train_set is not None, "TableDataLoader.normalize: call the 'split' method first"
        print(self.train_set.values)

if __name__ == "__main__":
    table_loader = TableDataLoader(data_dir = "../../../data/gravo/")
    table_loader.filter(stage = STAGE_PRETREATMENT)
    table_loader.split()
    # table_loader.normalize()
