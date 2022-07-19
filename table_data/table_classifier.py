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
        
    def compute_metrics(self, set: str):
        assert set, f"TableClassifier.get_predictions: Unknown set {set}. Available sets are {[s for s in self.sets]}"
        y_prob = self.get_predictions(self.sets[set]["x"])
        y_true = self.sets[set]["y"]
        import numpy as np
        l, c = np.unique(y_true, return_counts = True)
        print(l, c)
        return compute_metrics(y_true, y_prob)
