import pandas as pd
import numpy as np

class ClinicClassifier:
    def __init__(self, table_loader, variables: list):
        table_loader.filter(to_keep = variables)
        table_loader.split()
        table_loader.impute()
        train_x   = pd.concat([table_loader.train_set["x"], table_loader.val_set["x"]])
        train_y   = np.concat([table_loader.train_set["y"], table_loader.val_set["y"]])
        self.sets = {"train": {"x": train_x, "y": train_y}, "test": table_loader.test_set}
        self.preprocess("train")
        self.preprocess("test")
    
    @abstractmethod
    def preprocess(self, set: str):
        pass
        
    @abstractmethod
    def get_probabilities(self, rows):
        pass
    
