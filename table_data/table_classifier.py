import sys
sys.path.append("..")
from abc import ABC, abstractmethod
from utils.trainer import compute_metrics


class TableClassifier(ABC):
    def __init__(self, table_loader, **kwargs):
        self.sets = {}
        self.load_sets(table_loader, **kwargs)
    
    @abstractmethod    
    def load_sets(self, **kwargs):
        pass
        
    @abstractmethod
    def get_predictions(self, x):
        pass
    
    @abstractmethod
    def get_columns(self) -> list:
        pass
        
    def get_x(self, set: str):
        return self.sets[set][self.get_columns()].values
        
    def get_y(self, set: str):
        return self.sets[set]["binary_rankin"].values
        
    def compute_metrics(self, set: str):
        assert set, f"TableClassifier.get_predictions: Unknown set {set}. Available sets are {[s for s in self.sets]}"
        y_prob = self.get_predictions(self.get_x(set))
        y_true = self.get_y(set)
        return compute_metrics(y_true, y_prob)
