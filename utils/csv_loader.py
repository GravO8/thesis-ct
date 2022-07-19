from abc import ABC, abstractmethod

class CSVLoader:
    @abstractmethod
    def filter(self):
        pass
        
    @abstractmethod
    def split(self):
        pass
        
    @abstractmethod
    def amputate(self):
        pass
        
    @abstractmethod
    def impute(self):
        pass
        
    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def get_set(self):
        pass
        
    @abstractmethod
    def set_set(self):
        pass
        
    def split(self):
        for set in ("train", "val", "test"):
            self.sets[set] = self.set_set(set)
        del self.table_df
