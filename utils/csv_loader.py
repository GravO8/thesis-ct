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
