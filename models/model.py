import torch
from abc import ABC, abstractmethod
from final_mlp import FinalMLP

class Model(ABC, torch.nn.Module):
    def __init__(self, model: torch.nn.Module, out_features: int, model_name: str):
        torch.nn.Module.__init__(self)
        self.encoder    = encoder
        self.mlp        = FinalMLP(out_features)
        self.model_name = model_name
        
    @abstractmethod
    def process_input(self, x):
        pass 
    
    def forward(self, x):
        x = self.process_input(x)
        x = self.encoder(x)
        return self.mlp(x)
    
    @abstractmethod
    def to_dict(self):
        pass
        
    def get_name(self):
        return self.model_name
