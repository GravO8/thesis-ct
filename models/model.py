import torch
from abc import ABC, abstractmethod

class Model(ABC, torch.nn.Module):
    def __init__(self, model: torch.nn.Module, model_name):
        self.model      = model
        self.model_name = model_name
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
        
    def get_name(self):
        return self.model_name
