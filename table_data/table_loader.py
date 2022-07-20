import sys
sys.path.append("..")
from utils.csv_loader import CSVLoader
from datetime import datetime


STAGE_BASELINE     = "baseline"
STAGE_PRETREATMENT = "pretreatment"
STAGE_24H          = "24h"
STAGE_DISCHARGE    = "discharge"
STAGES             = (STAGE_BASELINE, STAGE_PRETREATMENT, STAGE_24H, STAGE_DISCHARGE)


def str_to_datetime(date: str):
    if (date is None) or (date == "None") or (date == "0"):
        return None
    return datetime.fromisoformat(date[:19])
    
def get_age(birthdate, day):
    if (birthdate is None) or (day is None):
        return None
    return day.year - birthdate.year - ((day.month, day.day) < (birthdate.month, birthdate.day))
    
def get_hour_delta(time1, time2):
    if (time1 is None) or (time2 is None) or (time2 < time1):
        return None
    return (time2 - time1).seconds // (60*60)
    
def convert_missing_to_nan(col):
    return np.array([np.nan if (v == "None") or (v is None) else v for v in col]).astype("float")
    

class TableLoader:
    def preprocess(self, ampute: bool = False, impute: bool = False, 
    filter_non_witnessed: bool = False, filter_out_no_ncct: bool = True):
        if filter_out_no_ncct:
            self.filter_no_ncct()
        self.add_vars()
        if impute: 
            assert not ampute
            self.impute()
        if ampute: 
            assert not impute
            self.ampute()
        if isinstance(keep_cols, str):
            keep_cols = self.stage_cols(keep_cols)
        return keep_cols
            
    def stage_cols(self, stage):
        to_remove.extend(["rankin-23", "NCCT", "CTA"])
        sections_to_remove = ["18"]
        if stage == STAGE_BASELINE:
            sections_to_remove.extend( ["7", "11", "15", "19", "20", "21", "22"] )
        elif stage == STAGE_PRETREATMENT:
            sections_to_remove.extend( ["11", "15", "19", "20", "21", "22"] )
        elif stage == STAGE_24H:
            sections_to_remove.extend( ["19", "20", "21", "22"] )
        elif stage != STAGE_DISCHARGE:
            assert False, f"TableLoader.stage_cols: the available are {STAGES}"
        for col in self.table.columns:
            for section in sections_to_remove:
                if col.endswith(section):
                    to_remove.append(col)
        return to_remove
        
    def filter_no_ncct(self):
        self.table = self.table[self.table["NCCT"] == "OK"]
        
    def add_vars(self, filter_non_witnessed: bool):
        '''
        add the variables 'age' and 'time_since_onset'
        'filter_non_witnessed' is a boolean that specifies whether to only consider
        patients whose stroke was witnessed in the 'time_since_onset' computation
        '''
        birthdate   = [str_to_datetime(date) for date in self.table["dataNascimento-1"].values]
        stroke_date = [str_to_datetime(date) for date in self.table["dataAVC-4"].values]
        ncct_time   = [str_to_datetime(date) for date in self.table["data-7"].values]
        age         = [get_age(birthdate[i],stroke_date[i]) for i in range(len(birthdate))]
        onset_time  = [get_hour_delta(stroke_date[i],ncct_time[i]) for i in range(len(birthdate))]
        if filter_non_witnessed:
            onset_time = [onset_time[i] if self.table["instAVCpre-4"].values[i]=="1" else None for i in range(len(birthdate))]
        self.table["age"]              = age
        self.table["time_since_onset"] = onset_time
