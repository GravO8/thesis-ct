import sys
sys.path.append("..")
from utils.table_data_loader import TableDataLoader
from utils.trainer import compute_metrics
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class ClinicClassifier:
    def __init__(self, table_loader, variables: list):
        table_loader.filter(to_keep = variables)
        table_loader.split()
        table_loader.impute()
        train_x   = pd.concat([table_loader.train_set["x"], table_loader.val_set["x"]])
        train_y   = np.concatenate([table_loader.train_set["y"], table_loader.val_set["y"]])
        self.sets = {"train": {"x": train_x, "y": train_y}, "test": table_loader.test_set}
        self.preprocess("train")
        self.preprocess("test")
    
    @abstractmethod
    def preprocess(self, set: str):
        pass
        
    @abstractmethod
    def get_probabilities(self, x):
        pass
        
    def compute_metrics(self, set: str):
        y_prob = self.get_probabilities(self.sets[set]["x"].values)
        y_true = self.sets[set]["y"]
        return compute_metrics(y_true, y_prob)
        
        
class ASTRALClinicClassifier(ClinicClassifier):
    def __init__(self, table_loader):
        super().__init__(table_loader, variables = ["age", "totalNIHSS-5", 
                                                    "time_since_onset", "gliceAd-4", 
                                                    "altVis-5", "altCons-5"])
    
    def preprocess(self, set: str):
        self.sets[set]["x"]["age"]              = self.sets[set]["x"]["age"].values // 5
        self.sets[set]["x"]["totalNIHSS-5"]     = self.sets[set]["x"]["totalNIHSS-5"].astype(int)
        self.sets[set]["x"]["time_since_onset"] = 2*(self.sets[set]["x"]["time_since_onset"].values > 3)
        self.sets[set]["x"]["altVis-5"]         = 2*(self.sets[set]["x"]["altVis-5"].values > 0)
        self.sets[set]["x"]["altCons-5"]        = 3*(self.sets[set]["x"]["altCons-5"].values > 0)
        glucose                                 = self.sets[set]["x"]["gliceAd-4"].values
        self.sets[set]["x"]["gliceAd-4"]        = (glucose > 131) | (glucose < 66)
        self.sets[set]["x"]                     = self.sets[set]["x"].astype(int)
        
    def get_probabilities(self, x):
        x = x.sum(axis = 1)
        x = x > 31
        return x.astype(int)
        
    
if __name__ == "__main__":
    table_loader = TableDataLoader(data_dir = "../../../data/gravo/")
    astral       = ASTRALClinicClassifier(table_loader)
    metrics      = astral.compute_metrics("train")
    print(metrics)
