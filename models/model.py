import torch
from abc import ABC, abstractmethod
from .models import final_mlp
from .encoder import Encoder
        
        
class Model(ABC, torch.nn.Module):
    def __init__(self, encoder: Encoder):
        torch.nn.Module.__init__(self)
        self.encoder    = encoder
        self.mlp        = final_mlp(self.encoder.out_features)
        
    @abstractmethod
    def process_input(self, x):
        pass
        
    @abstractmethod
    def name_appendix():
        pass
        
    def normalize_input(self, x, range_max: int = 100):
        return x/range_max
    
    def forward(self, x):
        x = self.process_input(x)
        x = self.encoder(x)
        return self.mlp(x)
        
    def get_name(self):
        return self.name_appendix() + "-" + self.encoder.get_name()


class Baseline3DCNN(Model):
    def process_input(self, x):
        return self.normalize_input(x)
    def name_appendix(self):
        return "baseline-3DCNN"
        
        
class BaselineMirror(Model):
    def process_input(self, x):
        x = self.normalize_input(x)
        x = x - x.flip(2)
        return x
    def name_appendix(self):
        return "baseline-mirror" 
