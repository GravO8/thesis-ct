import os
import pandas as pd
from datetime import datetime


def str_to_datetime(date: str):
    if (date is None) or (date == "None"):
        return None
    return datetime.fromisoformat(date[:19])


class TableDataLoader:
    def __init__(self, table_filename: str = "table_data.csv", 
    labels_filename: str = "dataset.csv", data_dir: str = None):
        data_dir        = "" if data_dir is None else data_dir
        table_filename  = os.path.join(data_dir, table_filename)
        labels_filename = os.path.join(data_dir, labels_filename)
        self.table_df   = pd.read_csv(table_filename)
        self.labels_df  = pd.read_csv(labels_filename)
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
        c = 0
        for d in stroke_date:
            if d is None:
                c+= 1
        print(c)
        # for _, row in self.table_df


if __name__ == "__main__":
    table_loader = TableDataLoader(data_dir = "../../../data/gravo/")
