import sys
sys.path.append("..")
from utils.table_data_loader import TableDataLoader
from utils.trainer import compute_metrics
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class ClinicClassifier:
    def __init__(self, table_loader):
        table_loader.filter(to_keep = self.get_columns())
        table_loader.split()
        # table_loader.impute()
        table_loader.amputate()
        train     = pd.concat([table_loader.train_set, table_loader.val_set])
        self.sets = {"train": train, "test": table_loader.test_set}
        self.preprocess("train")
        self.preprocess("test")
    
    @abstractmethod
    def preprocess(self, set: str):
        pass
        
    @abstractmethod
    def get_scores(self, x):
        pass
        
    @abstractmethod
    def get_probabilities(self, x):
        pass
    
    @abstractmethod
    def get_columns(self) -> list:
        pass
        
    def compute_metrics(self, set: str):
        y_prob = self.get_probabilities(self.sets[set][self.get_columns()].values)
        y_true = self.sets[set]["binary_rankin"]
        return compute_metrics(y_true, y_prob)
        
        
class ASTRALClinicClassifier(ClinicClassifier):
    def __init__(self, table_loader):
        super().__init__(table_loader)
    
    def preprocess(self, set: str):
        self.sets[set]["age"]              = self.sets[set]["age"].values // 5
        self.sets[set]["totalNIHSS-5"]     = self.sets[set]["totalNIHSS-5"].astype(int)
        self.sets[set]["time_since_onset"] = 2*(self.sets[set]["time_since_onset"].values > 3)
        self.sets[set]["altVis-5"]         = 2*(self.sets[set]["altVis-5"].values > 0)
        self.sets[set]["altCons-5"]        = 3*(self.sets[set]["altCons-5"].values > 0)
        glucose                            = self.sets[set]["gliceAd-4"].values
        self.sets[set]["gliceAd-4"]        = (glucose > 131) | (glucose < 66)
        self.sets[set]                     = self.sets[set].astype(int)
        
    def get_scores(self, x):
        return x.sum(axis = 1)
        
    def get_probabilities(self, x):
        x = self.get_scores(x)
        x = x > 31
        return x.astype(int)
        
    def get_columns(self):
        return ["age", "totalNIHSS-5", "time_since_onset", "altVis-5", "altCons-5", "gliceAd-4"]
        
    
if __name__ == "__main__":
    table_loader = TableDataLoader(data_dir = "../../../data/gravo/")
    astral       = ASTRALClinicClassifier(table_loader)
    # metrics      = astral.compute_metrics("train")
    # print(metrics)
    
    pred   = astral.get_probabilities( astral.sets["test"][astral.get_columns()].values )
    scores = astral.get_scores( astral.sets["test"][astral.get_columns()].values )
    astral.sets["test"]["pred"]   = pred
    astral.sets["test"]["astral"] = scores
    print(astral.sets["test"][["idProcessoLocal", "astral", "pred"]])
