import sys
sys.path.append("..")
from abc import ABC, abstractmethod
from utils.trainer import compute_metrics


class TableClassifier(ABC):
    def __init__(self):
        self.sets = {}
        
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
        assert set, f"TableClassifier.compute_metrics: Unknown set {set}. Available sets are {[s for s in self.sets]}"
        y_prob = self.get_probabilities(self.sets[set][self.get_columns()].values)
        y_true = self.sets[set]["binary_rankin"].values
        return compute_metrics(y_true, y_prob)
