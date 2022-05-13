import torch
from model_name import to_model_id, to_model_name
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, model: torch.nn.Module, model_id: str = None, model_name: str = None):
        assert (model_id is not None) or (model_name is not None)
        self.model      = model
        self.model_id   = model_id
        self.model_name = model_name
        
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
        
    def get_id(self):
        if self.model_id is None:
            self.model_id = to_model_id(self.model_name)
        return self.model_id
        
    def get_name(self):
        if self.model_name is None:
            self.model_name = to_model_name(self.model_id)
        return self.model_name
        
    def get_id_name(self):
        return self.get_id() + self.get_name()
